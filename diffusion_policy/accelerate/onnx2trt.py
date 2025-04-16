import tensorrt as trt

import numpy as np

# 定义日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    """
    从 ONNX 文件构建 TensorRT 引擎
    :param onnx_file_path: ONNX 文件路径
    :param engine_file_path: 保存引擎文件的路径
    :return: TensorRT 引擎
    """
    # 创建 TensorRT 构建器
    builder = trt.Builder(TRT_LOGGER)
    
    # 创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建 ONNX 解析器
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 加载并解析 ONNX 文件
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ONNX 解析失败")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 创建构建器配置
    config = builder.create_builder_config()
    
    # 设置工作空间大小（1GB）
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # 设置 FP16 精度（如果需要）
    # if builder.platform_has_fast_fp16:
    #     config.set_flag(trt.BuilderFlag.FP16)
    
    # 构建序列化的 TensorRT 引擎
    print("========Start serialize==============")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("构建序列化引擎失败")
        return None
    
    # 反序列化引擎
    print("========Start deserialize==============")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        print("反序列化引擎失败")
        return None
    
    # 保存引擎到文件
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    
    return engine

# 构建 TensorRT 引擎
engine = build_engine("unet1d_noInstance.onnx", "unet_fp32.engine")
if engine:
    print("TensorRT引擎已创建")
else:
    print("TensorRT引擎创建失败")



'''
ONNX to TensorRT Conversion Utility
This script converts an ONNX model to a TensorRT engine for faster inference.
TensorRT Overview:
-----------------
TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime.
It significantly improves inference performance on NVIDIA GPUs compared to
frameworks like PyTorch or TensorFlow.
Key Components:
-------------
1. Logger (TRT_LOGGER):
    - Controls verbosity of TensorRT messages (WARNING level used here)
2. Builder:
    - Main TensorRT object that creates optimization profiles and networks
3. Network:
    - Represents the model structure
    - EXPLICIT_BATCH flag allows for dynamic batch size support
4. OnnxParser:
    - Parses ONNX model into TensorRT's internal representation
5. BuilderConfig:
    - Contains settings for the optimization process
    - Controls memory usage, precision modes, etc.
6. OptimizationProfile:
    - Defines input shape ranges for models with dynamic shapes
    - Specifies min/optimal/max shapes for each input
7. Runtime:
    - Used to deserialize and execute the engine
Key Features Demonstrated:
------------------------
- Dynamic shape handling with optimization profiles
- FP16 precision enabling (when hardware supports it)
- Workspace memory allocation (1GB in this example)
- Engine serialization for later use without recompilation
Usage:
-----
1. Call build_engine() with paths to:
    - Input ONNX model file
    - Output TensorRT engine file (.engine)
2. The function returns the TensorRT engine object if successful, 
    or None if errors occur during conversion.
Notes for Beginners:
------------------
- TensorRT engines are platform-specific (must run on same GPU architecture)
- Larger workspace sizes allow more optimizations but consume more GPU memory
- Dynamic shapes allow flexibility but may reduce performance compared to fixed shapes
- FP16 mode significantly improves performance with minimal accuracy loss on modern GPUs

'''