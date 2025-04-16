if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    import torch
    import onnx
    import onnxsim
    import yaml
    import pprint
    import logging
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 加载YAML配置文件
    config_path = "/home/luna/dp/data/experiments/low_dim/square_ph/diffusion_policy_cnn/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("===== Configuration =====")
    policy_config = config['policy']
    model_config = policy_config['model']
    pprint.pprint(model_config)
    
    # 从配置中获取所需参数
    input_dim = model_config['input_dim']  # 10
    local_cond_dim = model_config['local_cond_dim']  # None
    global_cond_dim = model_config['global_cond_dim']  # 46
    down_dims = model_config['down_dims']  # [256, 512, 1024]
    horizon = policy_config['horizon']  # 16
    
    print(f"\nInput dim: {input_dim}")
    print(f"Local cond dim: {local_cond_dim}")
    print(f"Global cond dim: {global_cond_dim}")
    print(f"Down dims: {down_dims}")
    print(f"Horizon: {horizon}")
    
    # 导入必要的模块
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    
    # 1. 首先创建UNet模型
    print("\n===== Building UNet Model =====")
    unet_model = ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=local_cond_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=model_config['diffusion_step_embed_dim'],
        down_dims=down_dims,
        kernel_size=model_config['kernel_size'],
        n_groups=model_config['n_groups'],
        cond_predict_scale=model_config['cond_predict_scale']
    )
    
    # 2. 创建噪声调度器
    print("\n===== Creating Noise Scheduler =====")
    noise_scheduler_config = policy_config['noise_scheduler']
    noise_scheduler = DDPMScheduler(
        beta_start=noise_scheduler_config['beta_start'],
        beta_end=noise_scheduler_config['beta_end'],
        beta_schedule=noise_scheduler_config['beta_schedule'],
        num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
        clip_sample=noise_scheduler_config['clip_sample'],
        prediction_type=noise_scheduler_config['prediction_type'],
    )
    
    # 3. 根据配置构建策略模型
    print("\n===== Building Policy =====")
    policy = DiffusionUnetLowdimPolicy(
        model=unet_model,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=config['obs_dim'],  # 23
        action_dim=input_dim,  # 10
        n_action_steps=config['n_action_steps'],  # 8
        n_obs_steps=config['n_obs_steps'],  # 2
        obs_as_global_cond=config['obs_as_global_cond'],  # True
        obs_as_local_cond=config['obs_as_local_cond'],  # False
        num_inference_steps=policy_config.get('num_inference_steps', 100),
        pred_action_steps_only=policy_config.get('pred_action_steps_only', False),
        oa_step_convention=policy_config.get('oa_step_convention', True)
    )
    
    # 加载预训练权重
    checkpoint_path = "/home/luna/dp/data/experiments/low_dim/square_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=1750-test_mean_score=1.000.ckpt"
    print(f"\n===== Loading weights from {checkpoint_path} =====")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    unet_model = policy.model
    # model = policy.model
    unet_model.eval()

    batch_size = 4  # 原来是1，现在改为4
    # UNet的输入也需要修改为批次大小4
    sample = torch.randn(batch_size, horizon, input_dim)  # (4,T,D)格式
    timestep = torch.zeros(batch_size, dtype=torch.long)  # (4,)格式

    # 条件输入也需要修改批次大小
    dummy_local_cond = torch.zeros(batch_size, horizon, 1)  # 占位符，批次大小为4
    global_cond = torch.randn(batch_size, global_cond_dim)  # 实际使用的全局条件，批次大小为4

    local_cond = None  # 在forward调用中仍使用None
    
    # 测试UNet的前向传递
    print("\n===== Testing UNet forward pass =====")
    with torch.no_grad():
        # 使用None进行正常模型测试
        unet_output = unet_model(sample, timestep, local_cond=local_cond, global_cond=global_cond)
        print(f"UNet output shape: {unet_output.shape}")

    # 为ONNX导出创建包装器
    class UNetExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, sample, timestep, global_cond):
            # 对于ONNX导出，我们始终传递实际参数，但在forward内部忽略dummy_local_cond
            return self.model(sample, timestep, local_cond=None, global_cond=global_cond)

    wrapper = UNetExportWrapper(unet_model)

    # ONNX导出时使用包装器和dummy输入
    torch.onnx.export(
        wrapper,
        (sample, timestep, global_cond),
        "unet1d_noInstance.onnx",
        input_names=['sample', 'timestep', 'global_cond'],
        output_names=['output'],
        training=torch.onnx.TrainingMode.EVAL,
        # 移除dynamic_axes参数，使用固定批次大小
        opset_version=13
    )
    
    model_onnx = onnx.load("unet1d_noInstance.onnx")
    # model_simp, check = onnxsim.simplify(model_onnx, skipped_optimizers=True,skip_shape_inference=False)
    # onnx.save(model_simp, "unet1d_simplified.onnx")
    print("UNet ONNX模型已导出并简化")

    # 4. 测试ONNX模型，使用ONNX Runtime测试与PyTorch结果是否一致
    print("\n===== 测试ONNX模型 =====")

    import onnxruntime as ort
    import numpy as np
    # 创建ONNX Runtime会话
    print("加载ONNX模型...")
    ort_session = ort.InferenceSession("unet1d_noInstance.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # 准备输入数据
    ort_inputs = {
        'sample': sample.numpy(),
        'timestep': timestep.numpy(),
        'global_cond': global_cond.numpy()
    }
    
    # 再次使用PyTorch模型获取参考输出
    print("运行PyTorch模型...")
    with torch.no_grad():
        pytorch_output = unet_model(sample, timestep, local_cond=None, global_cond=global_cond)
        
    # 运行ONNX Runtime推理
    print("运行ONNX Runtime推理...")
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_output = ort_outputs[0]  # 第一个输出即为我们需要的结果
    
    # 比较结果
    pytorch_output_np = pytorch_output.numpy()
    
    # 检查形状是否一致
    print(f"PyTorch输出形状: {pytorch_output_np.shape}")
    print(f"ONNX输出形状: {ort_output.shape}")
    
    if pytorch_output_np.shape == ort_output.shape:
        # 计算差异
        abs_diff = np.abs(pytorch_output_np - ort_output)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        print(f"最大绝对误差: {max_diff}")
        print(f"平均绝对误差: {mean_diff}")
        
        # 计算相对误差
        pytorch_norm = np.linalg.norm(pytorch_output_np)
        if pytorch_norm > 0:
            rel_error = np.linalg.norm(ort_output - pytorch_output_np) / pytorch_norm
            print(f"相对误差: {rel_error:.6f}")
            
            if rel_error < 1e-4:
                print("✅ ONNX转换非常精确 - 可以安全使用ONNX模型")
            elif rel_error < 1e-2:
                print("⚠️ ONNX转换精度可接受 - 但请验证在您的应用中的影响")
            else:
                print("❌ ONNX转换精度较低 - 需要调查原因")
    else:
        print("❌ 输出形状不匹配!")

