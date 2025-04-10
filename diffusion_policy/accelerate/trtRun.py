import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # 分配输入/输出内存
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape) * 4  # 假设是FP32
            
            # 分配GPU内存
            allocation = cuda.mem_alloc(size)
            
            binding = {
                'index': i,
                'name': name,
                'dtype': dtype,
                'shape': shape,
                'allocation': allocation,
                'is_input': is_input
            }
            
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
    
    def predict(self, sample, timestep, local_cond, global_cond):
        # 将输入数据复制到GPU
        cuda.memcpy_htod(self.inputs[0]['allocation'], sample.astype(np.float32).ravel())
        cuda.memcpy_htod(self.inputs[1]['allocation'], timestep.astype(np.int64).ravel())
        cuda.memcpy_htod(self.inputs[2]['allocation'], local_cond.astype(np.float32).ravel())
        cuda.memcpy_htod(self.inputs[3]['allocation'], global_cond.astype(np.float32).ravel())
        
        # 执行推理
        self.context.execute_v2(self.allocations)
        
        # 获取输出
        output = np.zeros(self.outputs[0]['shape'], dtype=np.float32)
        cuda.memcpy_dtoh(output, self.outputs[0]['allocation'])
        
        return output

# 使用TensorRT模型进行推理
trt_model = TRTModel("unet1d.engine")

# 准备输入数据
batch_size = 1
horizon = 32
input_dim = 32
local_cond_dim = 64
global_cond_dim = 128

sample_np = np.random.randn(batch_size, input_dim, horizon).astype(np.float32)
timestep_np = np.array([0], dtype=np.int64)
local_cond_np = np.random.randn(batch_size, local_cond_dim, horizon).astype(np.float32)
global_cond_np = np.random.randn(batch_size, global_cond_dim).astype(np.float32)

# 执行推理
output = trt_model.predict(sample_np, timestep_np, local_cond_np, global_cond_np)
print("TensorRT推理完成，输出形状:", output.shape)