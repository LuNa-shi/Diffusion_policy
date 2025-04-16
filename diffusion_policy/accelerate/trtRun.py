import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

class TRTModel:
    def __init__(self, engine_path):
        # 使用TensorRT 8.6.1兼容的logger创建
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 分配输入/输出内存
        self.inputs = []
        self.outputs = []
        self.bindings = []  # 用于存储绑定指针
        
        # 输出绑定信息
        print("模型绑定信息:")
        for i in range(self.engine.num_io_tensors):  
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)  
            shape = self.engine.get_tensor_shape(name)  
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT  
            
            print(f"Tensor {i}: {name}, {'输入' if is_input else '输出'}, 形状: {shape}, 类型: {dtype}")
            
            # 计算内存大小
            original_shape = shape
            # 处理动态维度（-1表示动态批量大小）
            if shape[0] == -1:
                # 创建形状的副本，将动态维度替换为合理的最大值
                modified_shape = list(shape)
                modified_shape[0] = 16  # 设置一个合理的最大批次大小，如16
                size = trt.volume(modified_shape)
                print(f"处理动态形状: 原始={original_shape}, 计算用={modified_shape}, 内存大小={size}")
            else:
                size = trt.volume(shape)
                print(f"内存大小: {size}")
            
            if dtype == trt.int64:
                size *= 8  # int64 = 8字节
            else:
                size *= 4  # 假设其他类型是float32 = 4字节
            
            # 分配GPU内存
            allocation = cuda.mem_alloc(size)
            
            binding = {
                'index': i,
                'name': name,
                'dtype': dtype,
                'shape': shape,  # 保留原始形状（含-1）
                'allocation': allocation,
                'is_input': is_input
            }
            
            self.bindings.append(int(allocation))
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
    
    def predict(self, inputs_dict):
        """
        执行推理
        inputs_dict: 字典，键为输入名称，值为numpy数组
        """
        # 将输入数据复制到GPU
        for input_binding in self.inputs:
            name = input_binding['name']
            if name in inputs_dict:
                data = inputs_dict[name]
                
                # 确保数据类型正确
                if input_binding['dtype'] == trt.int64:
                    data = data.astype(np.int64)
                else:
                    data = data.astype(np.float32)
                
                # 检查形状是否匹配
                if data.shape != input_binding['shape']:
                    print(f"警告: 输入 {name} 形状不匹配。预期: {input_binding['shape']}, 实际: {data.shape}")
                    # 尝试调整形状
                    if len(data.shape) == len(input_binding['shape']):
                        # 使用新API设置输入形状
                        self.context.set_input_shape(name, data.shape)
                        print(f"已设置 {name} 的动态形状为 {data.shape}")
                
                cuda.memcpy_htod(input_binding['allocation'], data.ravel())
            else:
                print(f"错误: 缺少输入 {name}")
                return None
        
        # 执行推理 - 使用新的推理API
        self.context.execute_async_v3(stream_handle=0)  # 使用默认流
        
        # 获取输出
        outputs = {}
        for output_binding in self.outputs:
            name = output_binding['name']
            # 使用新API获取输出形状
            actual_shape = self.context.get_tensor_shape(name)
            
            output = np.empty(actual_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, output_binding['allocation'])
            outputs[name] = output
        
        return outputs


def main():
    # 使用TensorRT模型进行推理
    print("加载TensorRT引擎...")
    trt_model = TRTModel("unet1d.engine")
    
    # 根据torch2onnx.py文件输出和ONNX模型的格式准备输入数据
    batch_size = 1
    horizon = 16    # 根据config: 16
    input_dim = 10  # 根据config: 10
    global_cond_dim = 46  # 根据config: 46
    
    # 注意：所有输入形状应该与ONNX导出中使用的形状一致
    # sample: [batch_size, horizon, input_dim] - ConditionalUnet1D的文档中指定
    sample = np.random.randn(batch_size, horizon, input_dim).astype(np.float32)
    
    # timestep: [batch_size] - 整数张量
    timestep = np.zeros(batch_size, dtype=np.int64)
    
    # dummy_local_cond: [batch_size, horizon, 1] - 虽然model不使用，但ONNX需要
    dummy_local_cond = np.zeros((batch_size, horizon, 1), dtype=np.float32)
    
    # global_cond: [batch_size, global_cond_dim]
    global_cond = np.random.randn(batch_size, global_cond_dim).astype(np.float32)
    
    # 准备输入字典，键名应与ONNX模型中的输入名相匹配
    inputs = {
        'sample': sample,
        'timestep': timestep,
        'local_cond': dummy_local_cond,
        'global_cond': global_cond
    }
    
    # 测量推理时间
    print("\n开始TensorRT推理...")
    start_time = time.time()
    for _ in range(10):  # 预热
        outputs = trt_model.predict(inputs)
    
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        outputs = trt_model.predict(inputs)
    elapsed = (time.time() - start_time) / num_runs * 1000  # 毫秒
    
    # 打印结果
    print(f"\nTensorRT推理完成, 平均耗时: {elapsed:.2f} ms")
    
    for name, output in outputs.items():
        print(f"输出 {name} 形状: {output.shape}")


if __name__ == "__main__":
    main()