import time
import torch

# 加载PyTorch模型
model = ConditionalUnet1D(
    input_dim=32,
    local_cond_dim=64,
    global_cond_dim=128,
    down_dims=[256, 512, 1024]
)
model.load_state_dict(torch.load("your_model_path.pth"))
model.eval()
model.cuda()

# 创建测试输入
sample = torch.randn(batch_size, input_dim, horizon).cuda()
timestep = torch.tensor([0], dtype=torch.long).cuda()
local_cond = torch.randn(batch_size, local_cond_dim, horizon).cuda()
global_cond = torch.randn(batch_size, global_cond_dim).cuda()

# PyTorch推理时间测试
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(100):  # 重复100次以获得更准确的测量
        output_torch = model(sample, timestep, local_cond, global_cond)
torch.cuda.synchronize()
pytorch_time = (time.time() - start) / 100
print(f"PyTorch平均推理时间: {pytorch_time*1000:.2f} ms")

# TensorRT推理时间测试
start = time.time()
for _ in range(100):  # 重复100次
    output_trt = trt_model.predict(
        sample_np, timestep_np, local_cond_np, global_cond_np)
trt_time = (time.time() - start) / 100
print(f"TensorRT平均推理时间: {trt_time*1000:.2f} ms")
print(f"加速比: {pytorch_time / trt_time:.2f}x")