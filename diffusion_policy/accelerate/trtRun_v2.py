import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os


class TRTModel:
    def __init__(self, engine_path):
        # 首先检查引擎文件是否存在
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"引擎文件未找到: {engine_path}")
        
        # 使用TensorRT 8.6.1兼容的logger
        self.logger = trt.Logger(trt.Logger.INFO)
        
        # 加载引擎
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
                if len(engine_data) == 0:
                    raise ValueError(f"引擎文件为空: {engine_path}")
                
                print(f"读取引擎文件: {engine_path}, 大小: {len(engine_data)} bytes")
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(engine_data)
                
                if self.engine is None:
                    raise RuntimeError("引擎反序列化失败")
                
                print(f"引擎反序列化成功, 张量数量: {self.engine.num_io_tensors}")
        except Exception as e:
            print(f"加载引擎时出错: {str(e)}")
            raise
            
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 初始化输入/输出
        self.inputs = []
        self.outputs = []

        print("模型绑定信息:")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            print(f"Tensor {i}: {name}, {'输入' if is_input else '输出'}, 形状: {shape}, 类型: {dtype}")

            # 计算内存分配大小
            max_shape = list(shape)
            found_dynamic = False
            for dim_idx, dim_val in enumerate(max_shape):
                if dim_val == -1:
                    if dim_idx == 0:
                        max_shape[dim_idx] = 16  # 假设最大批次大小为16
                        found_dynamic = True
                        print(f"  检测到动态维度 {dim_idx}，使用最大值 {max_shape[dim_idx]} 进行内存分配")
                    else:
                        raise ValueError(f"需要为张量 {name} 的动态维度 {dim_idx} 提供最大值以分配内存")

            size = trt.volume(max_shape) * trt.DataType(dtype).itemsize
            print(f"  计算内存大小基于形状: {max_shape}, 大小: {size} bytes")
            
            # 分配GPU内存
            allocation = cuda.mem_alloc(size)

            binding_info = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': tuple(shape),
                'max_shape': tuple(max_shape),
                'allocation': allocation,
                'is_input': is_input,
                'size': size  # 保存分配的大小以便后续检查
            }

            if is_input:
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)

    def predict(self, inputs_dict):
        # 1. 设置输入形状并验证
        for input_binding in self.inputs:
            name = input_binding['name']
            if name in inputs_dict:
                data = inputs_dict[name]

                # 确保数据类型正确
                if data.dtype != input_binding['dtype']:
                    data = data.astype(input_binding['dtype'])

                # 检查形状是否匹配引擎定义
                engine_shape = input_binding['shape']
                actual_shape = data.shape

                # 检查维度数量
                if len(actual_shape) != len(engine_shape):
                    print(f"错误: 输入 {name} 维度数量不匹配。预期: {len(engine_shape)}, 实际: {len(actual_shape)}")
                    return None

                is_dynamic = any(d == -1 for d in engine_shape)

                if is_dynamic:
                    # 检查尺寸是否在允许范围内
                    max_shape = input_binding['max_shape']
                    for i, (actual, max_val) in enumerate(zip(actual_shape, max_shape)):
                        if actual > max_val:
                            print(f"错误: 输入 {name} 维度 {i} 超出最大值。实际: {actual}, 最大允许: {max_val}")
                            return None
                    
                    # 设置实际输入形状
                    self.context.set_input_shape(name, actual_shape)
                elif actual_shape != engine_shape:
                    print(f"错误: 输入 {name} 形状不匹配。预期: {engine_shape}, 实际: {actual_shape}")
                    return None

                # 复制数据到GPU
                cuda.memcpy_htod(input_binding['allocation'], data.ravel())
            else:
                print(f"错误: 缺少输入 {name}")
                return None

        # 2. 设置所有张量的地址
        for binding in self.inputs:
            self.context.set_tensor_address(binding['name'], int(binding['allocation']))
        for binding in self.outputs:
            self.context.set_tensor_address(binding['name'], int(binding['allocation']))

        # 3. 执行推理
        try:
            status = self.context.execute_async_v3(stream_handle=0)
            if not status:
                print("推理执行失败!")
                return None
                
            # 同步流，确保计算完成
            cuda.Stream(0).synchronize()
        except Exception as e:
            print(f"执行推理时出错: {str(e)}")
            return None

        # 4. 获取输出
        outputs = {}
        for output_binding in self.outputs:
            name = output_binding['name']
            # 获取本次推理的实际输出形状
            actual_shape = self.context.get_tensor_shape(name)
            
            # 创建输出数组并复制数据
            output = np.empty(actual_shape, dtype=output_binding['dtype'])
            cuda.memcpy_dtoh(output, output_binding['allocation'])
            outputs[name] = output

        return outputs

    def cleanup(self):
        print("释放GPU内存...")
        for binding in self.inputs + self.outputs:
            try:
                binding['allocation'].free()
                print(f"  已释放 {binding['name']} 的内存")
            except Exception as e:
                print(f"  释放 {binding['name']} 内存时出错: {e}")
        print("清理完成。")


def main():
    # 使用更明确的文件路径
    engine_path = "unet_fp32.engine"  # 确保与生成的引擎文件名匹配
    trt.init_libnvinfer_plugins(None,'')
    try:
        print(f"加载TensorRT引擎: {engine_path}")
        trt_model = TRTModel(engine_path)
        
        try:
            # 准备输入数据
            batch_size = 4  # 使用与ONNX导出时相同的批次大小
            horizon = 16
            input_dim = 10
            global_cond_dim = 46

            sample = np.random.randn(batch_size, horizon, input_dim).astype(np.float32)
            timestep = np.zeros(batch_size, dtype=np.int64)
            dummy_local_cond = np.zeros((batch_size, horizon, 1), dtype=np.float32)
            global_cond = np.random.randn(batch_size, global_cond_dim).astype(np.float32)

            inputs = {
                'sample': sample,
                'timestep': timestep,
                'local_cond': dummy_local_cond,  # 如果ONNX导出时包含了这个输入
                'global_cond': global_cond
            }

            # 预热运行
            print("\n开始TensorRT预热 (10次)...")
            for i in range(10):
                outputs = trt_model.predict(inputs)
                if outputs is None:
                    print(f"预热第 {i+1} 次失败")
                    return

            # 性能测试
            print("\n开始TensorRT性能测试 (100次)...")
            num_runs = 100
            start_time = time.time()
            for i in range(num_runs):
                outputs = trt_model.predict(inputs)
                if outputs is None:
                    print(f"测试第 {i+1} 次失败")
                    return
            elapsed = (time.time() - start_time) / num_runs * 1000  # 毫秒

            # 输出结果
            print(f"\nTensorRT推理完成, 平均耗时: {elapsed:.2f} ms")
            print(f"每秒推理次数: {1000/elapsed:.1f}")

            for name, output in outputs.items():
                print(f"输出 {name} 形状: {output.shape}")
                print(f"输出 {name} 范围: [{output.min():.4f}, {output.max():.4f}]")

        finally:
            # 确保清理内存
            trt_model.cleanup()
            
    except Exception as e:
        print(f"运行过程中出错: {e}")


if __name__ == "__main__":
    main()