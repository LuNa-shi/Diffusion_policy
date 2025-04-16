import onnx

def count_parameters(model_path):
    # 加载 ONNX 模型
    model = onnx.load(model_path)
    
    # 计算参数量
    total_params = 0
    for initializer in model.graph.initializer:
        shape = initializer.dims
        params = 1
        for dim in shape:
            params *= dim
        total_params += params
    
    return total_params
def count_layers(model_path):
    # 加载 ONNX 模型
    model = onnx.load(model_path)
    
    # 计算层数（节点数量）
    return len(model.graph.node)


import onnxruntime as ort
import numpy as np
import torch
import time

def benchmark_model(model_path, input_data, warmup=10, repeat=100):
    # 创建 ONNX Runtime 会话

    model = onnx.load(model_path)
    # 打印模型输入
    print("模型输入:")
    for input in model.graph.input:
        print(f"Name: {input.name}, Shape: {input.type.tensor_type.shape}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Warmup
    for _ in range(warmup):
        session.run(None, input_data)
    
    # Benchmark
    total_time = 0
    for _ in range(repeat):
        start = time.time()
        session.run(None, input_data)
        total_time += time.time() - start
    
    # 计算平均推理时间
    avg_time = total_time / repeat * 1000  # 转换为毫秒
    return avg_time


# 定义输入维度
batch_size = 1  # 示例 batch size
horizon = 16    # 示例序列长度
input_dim = 10  # 示例输入维度
global_cond_dim = 46  # 示例全局条件维度
local_cond_dim = None   # 示例局部条件维度

# UNet的输入，注意格式应为(batch_size, horizon, input_dim)而不是(batch_size, input_dim, horizon)
sample = torch.randn(batch_size, horizon, input_dim)  # 修正为(B,T,D)格式
timestep = torch.zeros(batch_size, dtype=torch.long)  # 使用(B,)格式而不是单个值

# 构建条件输入，同样注意格式
global_cond = None
if global_cond_dim is not None:
    global_cond = torch.randn(batch_size, global_cond_dim)  # 这个格式是正确的(B,D)

local_cond = None
if local_cond_dim is not None:
    # 注意local_cond应该是(B,T,D)格式，而不是(B,D,T)
    local_cond = torch.randn(batch_size, horizon, local_cond_dim)

print(f"local_cond shape: {local_cond.shape if local_cond is not None else None}")
print(f"global_cond shape: {global_cond.shape if global_cond is not None else None}")
# 准备输入数据
input_data = {
    "sample": sample.numpy(),
    "timestep": timestep.numpy(),
    "local_cond": global_cond.numpy() if local_cond is not None else None,
    "global_cond": global_cond.numpy() if global_cond is not None else None,
}

# 过滤掉 None 输入
# input_data = {k: v for k, v in input_data.items() if v is not None}

# 测试未简化模型
# original_time = benchmark_model("unet1d.onnx", input_data)
# print(f"未简化模型的平均推理时间: {original_time:.2f} ms")


# # 对比性能
# print(f"推理时间变化: {simplified_time - original_time:.2f} ms")

# # 计算未简化模型的层数
# original_layers = count_layers("unet1d.onnx")
# print(f"未简化模型的层数: {original_layers}")


# # 对比层数
# print(f"层数变化: {simplified_layers - original_layers}")


# 计算未简化模型的参数量
original_params = count_parameters("unet1d.onnx")
print(f"未简化模型的参数量: {original_params}")

