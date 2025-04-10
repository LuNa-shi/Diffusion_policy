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

    batch_size = 1
    # UNet的输入
    sample = torch.randn(batch_size, horizon, input_dim)  # (B,T,D)格式
    timestep = torch.zeros(batch_size, dtype=torch.long)  # (B,)格式

    # 始终为两个条件创建tensor，确保ONNX导出中包含这两个输入
    dummy_local_cond = torch.zeros(batch_size, horizon, 1)  # 占位符
    global_cond = torch.randn(batch_size, global_cond_dim)  # 实际使用的全局条件

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
            
        def forward(self, sample, timestep, dummy_local_cond, global_cond):
            # 对于ONNX导出，我们始终传递实际参数，但在forward内部忽略dummy_local_cond
            return self.model(sample, timestep, local_cond=None, global_cond=global_cond)

    wrapper = UNetExportWrapper(unet_model)

    # ONNX导出时使用包装器和dummy输入
    torch.onnx.export(
        wrapper,
        (sample, timestep, dummy_local_cond, global_cond),  # 使用dummy_local_cond确保ONNX中有这个输入
        "unet1d.onnx",
        input_names=['sample', 'timestep', 'local_cond', 'global_cond'],
        output_names=['output'],
        dynamic_axes={
            'sample': {0: 'batch_size'},
            'timestep': {0: 'batch_size'},
            'local_cond': {0: 'batch_size'},
            'global_cond': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=13

    )
    
    model_onnx = onnx.load("unet1d.onnx")
    # model_simp, check = onnxsim.simplify(model_onnx, skipped_optimizers=True,skip_shape_inference=False)
    # onnx.save(model_simp, "unet1d_simplified.onnx")
    print("UNet ONNX模型已导出并简化")