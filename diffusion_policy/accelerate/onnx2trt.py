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
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 创建优化配置文件
    profile = builder.create_optimization_profile()
    
    # 遍历网络输入，设置动态形状范围
    for i in range(network.num_inputs):
        input = network.get_input(i)
        input_name = input.name
        input_shape = input.shape  # 输入形状 (batch_size, ...)
        
        # 设置动态形状范围
        if -1 in input_shape:  # 如果输入是动态形状
            min_shape = (1, *input_shape[1:])  # 最小形状
            opt_shape = (4, *input_shape[1:])  # 最优形状
            max_shape = (8, *input_shape[1:])  # 最大形状
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    
    # 将优化配置文件添加到构建器配置中
    config.add_optimization_profile(profile)
    
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
engine = build_engine("unet1d.onnx", "unet.engine")
if engine:
    print("TensorRT引擎已创建")
else:
    print("TensorRT引擎创建失败")