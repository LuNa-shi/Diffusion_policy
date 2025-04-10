import torch
import pprint

def inspect_checkpoint(ckpt_path):
    # 加载checkpoint
    checkpoint = torch.load(ckpt_path)
    
    # 检查checkpoint结构
    print("===== Checkpoint Keys =====")
    for key in checkpoint.keys():
        print(f"- {key}")
    
    # 如果有state_dicts，检查模型结构
    if "state_dicts" in checkpoint:
        print("\n===== Model Structure =====")
        if "model" in checkpoint["state_dicts"]:
            model_state = checkpoint["state_dicts"]["model"]
            
            # 提取模型参数信息
            model_info = {}
            for key in model_state.keys():
                # 查找关键参数
                if 'local_cond_encoder' in key and '0.blocks.0.conv.weight' in key:
                    # 从本层权重推断local_cond_dim (输入通道数)
                    model_info['local_cond_dim'] = model_state[key].shape[1]
                
                if 'down_modules.0.0.blocks.0.conv.weight' in key:
                    # 第一个下采样层的输入通道数就是input_dim
                    model_info['input_dim'] = model_state[key].shape[1]
                
                if 'down_modules.0.0.blocks.0.conv.weight' in key:
                    # 第一个下采样层的输出通道数是down_dims[0]
                    model_info['down_dims_0'] = model_state[key].shape[0]
                
                if 'down_modules.1.0.blocks.0.conv.weight' in key:
                    # 第二个下采样层的输出通道数是down_dims[1]
                    model_info['down_dims_1'] = model_state[key].shape[0]
                
                if 'down_modules.2.0.blocks.0.conv.weight' in key:
                    # 第三个下采样层的输出通道数是down_dims[2]
                    model_info['down_dims_2'] = model_state[key].shape[0]
                
                # 查找global_cond_dim
                if 'cond_encoder.1.weight' in key:
                    # 从输入维度推断global_cond_dim
                    total_cond_dim = model_state[key].shape[1]
                    # 总条件维度 = diffusion_step_embed_dim + global_cond_dim
                    # 默认diffusion_step_embed_dim为256
                    model_info['possible_global_cond_dim'] = total_cond_dim - 256
                    
            print("Inferred model parameters:")
            pprint.pprint(model_info)
                    
            print(f"\nTotal parameters: {sum(p.numel() for p in model_state.values())}")
            
        # 打印一些示例张量形状
        print("\n===== Sample Tensor Shapes =====")
        for key, value in list(model_state.items())[:10]:  # 只打印前10个
            print(f"{key}: {value.shape}")
    
    # 检查超参数/配置
    if "hyper_parameters" in checkpoint:
        print("\n===== Hyper Parameters =====")
        pprint.pprint(checkpoint["hyper_parameters"])
        
    return checkpoint

# 使用方式
ckpt_path = "/home/luna/dp/data/experiments/low_dim/square_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=1750-test_mean_score=1.000.ckpt"
checkpoint = inspect_checkpoint(ckpt_path)