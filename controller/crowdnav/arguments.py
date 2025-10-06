"""
CrowdNav参数配置 - 简化版本用于测试
"""
import torch

def get_crowdnav_args():
    """获取CrowdNav默认参数"""
    class Args:
        def __init__(self):
            # 网络参数
            self.human_node_rnn_size = 128
            self.human_human_edge_rnn_size = 256
            self.human_node_input_size = 3
            self.human_human_edge_input_size = 2
            self.human_node_output_size = 256
            self.human_node_embedding_size = 64
            self.human_human_edge_embedding_size = 64
            self.attention_size = 64
            self.seq_length = 30
            
            # 注意力参数
            self.use_self_attn = True
            self.use_hr_attn = True
            self.sort_humans = True
            
            # 环境参数
            self.env_name = 'your-custom-env-name'  # 使用你的自定义环境
            
            # CUDA设置
            self.no_cuda = not torch.cuda.is_available()
            self.cuda = torch.cuda.is_available()
            
            # 训练参数
            self.num_processes = 1
            self.num_mini_batch = 2
            self.seed = 1
            self.gamma = 0.99
            self.cuda_deterministic = False
            
            # 其他参数
            self.recurrent_policy = True
    
    return Args()

def get_args():
    """兼容性函数，调用get_crowdnav_args"""
    return get_crowdnav_args()