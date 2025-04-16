import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

def compare_outputs(onnx_output, trt_output, atol=1e-3, rtol=1e-3):
    """比较两个numpy数组是否在允许误差范围内一致"""
    print("\n===== 输出对比结果 =====")
    print(f"ONNX输出形状：{onnx_output.shape} 数据类型：{onnx_output.dtype}")
    print(f"TRT输出形状：{trt_output.shape} 数据类型：{trt_output.dtype}")
    
    # 使用numpy的allclose函数进行精度比较
    is_close = np.allclose(onnx_output, trt_output, atol=atol, rtol=rtol)
    diff = np.abs(onnx_output - trt_output)
    
    print(f"最大绝对误差：{np.max(diff):.2e}")
    print(f"平均绝对误差：{np.mean(diff):.2e}")
    print(f"验证结果：{'通过✅' if is_close else '失败❌'}")
    return is_close

def benchmark_onnx_inference(onnx_path, inputs_dict, num_warmup=10, num_runs=100):
    """运行ONNX推理性能测试"""
    print("\n===== 运行ONNX推理 =====")
    session = ort.InferenceSession(onnx_path)
    
    # 动态获取输入名称
    input_names = [inp.name for inp in session.get_inputs()]
    print("检测到的输入名称:", input_names)
    
    # 预热阶段
    print(f"预热中 ({num_warmup} 次)...")
    for _ in range(num_warmup):
        _ = session.run(None, inputs_dict)
    
    # 性能测试
    print(f"性能测试中 ({num_runs} 次)...")
    start_time = time.time()
    for _ in range(num_runs):
        outputs = session.run(None, inputs_dict)
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / num_runs * 1000  # 毫秒
    print(f"ONNX Runtime 平均推理时间: {elapsed_time:.2f} ms")
    print(f"ONNX Runtime 吞吐量: {1000/elapsed_time:.1f} 推理/秒")
    
    return outputs[0], elapsed_time  # 返回第一个输出和推理时间

def benchmark_trt_inference(trt_path, inputs_dict, num_warmup=10, num_runs=100):
    """运行TensorRT推理性能测试（适配TensorRT 8.6.1 API）"""
    print("\n===== 运行TensorRT推理 =====")
    # 加载TRT引擎
    with open(trt_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    # 创建执行上下文
    context = engine.create_execution_context()
    
    # 准备绑定和缓冲区
    bindings = []
    buffers = {}
    
    # 获取所有张量名称并区分输入/输出
    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
    
    # 处理输入
    for name in input_names:
        # 从输入字典获取数据
        data = inputs_dict[name].astype(np.float32)
        buffers[name] = cuda.mem_alloc(data.nbytes)
        cuda.memcpy_htod(buffers[name], data)
        context.set_tensor_address(name, int(buffers[name]))
        
        # 设置输入形状（如果是动态形状）
        if -1 in engine.get_tensor_shape(name):
            context.set_input_shape(name, data.shape)
    
    # 预处理输出内存分配
    outputs = {}
    for name in output_names:
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = context.get_tensor_shape(name)
        buffers[name] = cuda.mem_alloc(np.zeros(shape, dtype).nbytes)
        context.set_tensor_address(name, int(buffers[name]))
        outputs[name] = np.empty(shape, dtype)
    
    # 创建CUDA流用于同步
    stream = cuda.Stream()
    
    # 预热
    print(f"预热中 ({num_warmup} 次)...")
    for _ in range(num_warmup):
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    
    # 性能测试
    print(f"性能测试中 ({num_runs} 次)...")
    start_time = time.time()
    for _ in range(num_runs):
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / num_runs * 1000  # 毫秒
    print(f"TensorRT 平均推理时间: {elapsed_time:.2f} ms")
    print(f"TensorRT 吞吐量: {1000/elapsed_time:.1f} 推理/秒")
    
    # 获取最终输出数据
    for name in output_names:
        cuda.memcpy_dtoh(outputs[name], buffers[name])
    
    # 释放资源
    for buffer in buffers.values():
        buffer.free()
    
    return outputs[output_names[0]], elapsed_time  # 返回第一个输出和推理时间

# 主验证流程
if __name__ == "__main__":
    # 初始化TensorRT插件
    trt.init_libnvinfer_plugins(None, '')
    
    # 文件路径配置
    onnx_path = "unet1d_noInstance.onnx"
    trt_engine_path = "unet_fp32.engine"
    
    # 创建测试输入（根据你的模型配置调整）
    batch_size = 4
    sample = np.random.randn(batch_size, 16, 10).astype(np.float32)  # [4, 16, 10]
    timestep = np.zeros(batch_size, dtype=np.int64)                  # [4]
    global_cond = np.random.randn(4, 46).astype(np.float32)          # [4, 46]
    
    # 构建输入字典（输入名称需要与模型定义一致）
    inputs_dict = {
        "sample": sample,
        "timestep": timestep,
        "global_cond": global_cond
    }
    
    # 运行推理基准测试
    onnx_output, onnx_time = benchmark_onnx_inference(onnx_path, inputs_dict)
    trt_output, trt_time = benchmark_trt_inference(trt_engine_path, inputs_dict)
    
    # 输出对比
    compare_outputs(onnx_output, trt_output)
    
    # 性能对比
    print("\n===== 性能对比 =====")
    print(f"ONNX Runtime: {onnx_time:.2f} ms")
    print(f"TensorRT:     {trt_time:.2f} ms")
    
    if trt_time < onnx_time:
        speedup = onnx_time / trt_time
        print(f"TensorRT 加速比: {speedup:.2f}x 更快")
    else:
        slowdown = trt_time / onnx_time
        print(f"TensorRT 速度下降: {slowdown:.2f}x 更慢")
    
    # 可视化性能对比
    print("\n性能对比图 (毫秒)")
    max_time = max(onnx_time, trt_time)
    onnx_bar = "=" * int(onnx_time * 50 / max_time)
    trt_bar = "=" * int(trt_time * 50 / max_time)
    
    print(f"ONNX: {onnx_bar} {onnx_time:.2f} ms")
    print(f"TRT : {trt_bar} {trt_time:.2f} ms")