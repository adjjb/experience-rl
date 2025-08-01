# gradient_bank.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from scipy.spatial.distance import cdist
from torch.distributions import Normal

class GradientBank(nn.Module):
    """基于注意力机制的梯度经验存储模块"""
    def __init__(self, action_dim, state_dim, device, 
                 capacity=1000, 
                 min_intervention=0, 
                 max_intervention=0.01,
                 intervention_steps=200000,
                 grad_clip=1.0):
        """
        初始化梯度经验存储模块
        
        参数:
            action_dim (int): 动作空间的维度
            state_dim (int): 状态表示的维度
            device (torch.device): 计算设备
            capacity (int): 经验缓冲区容量
            min_intervention (float): 最小干预强度
            max_intervention (float): 最大干预强度
            intervention_steps (int): 干预强度达到最大所需的步数
            grad_clip (float): 梯度裁剪阈值
        """
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        self.capacity = capacity
        self.grad_clip = grad_clip
        self.min_intervention = min_intervention
        self.max_intervention = max_intervention
        self.intervention_steps = intervention_steps
        
        # 注意力机制参数
        self.query_net = nn.Linear(state_dim, 64).to(device)
        self.key_net = nn.Linear(state_dim, 64).to(device)
        
        # 经验缓冲区
        self.state_buffer = deque(maxlen=capacity)
        self.grad_buffer = deque(maxlen=capacity)
        
        # 经验向量初始化
        self.experience = nn.Parameter(torch.zeros(1, action_dim, device=device), requires_grad=False)
        self.n_updates = 0
        
    def update(self, state_rep, new_grad):
        """使用注意力机制更新经验向量"""
        # 梯度裁剪
        if self.grad_clip is not None:
            new_grad = torch.clamp(new_grad, -self.grad_clip, self.grad_clip)
            
        # 添加到缓冲区 - 确保存储的是展平的一维向量
        state_rep = state_rep.squeeze(0)  # 移除批次维度
        new_grad = new_grad.squeeze(0)    # 移除批次维度
        
        self.state_buffer.append(state_rep.detach().cpu().numpy())
        self.grad_buffer.append(new_grad.detach().cpu().numpy())
        
        # 更新全局经验向量
        self._update_global_experience()
        self.n_updates += 1
        
    def _update_global_experience(self):
        """使用注意力机制更新全局经验向量"""
        if len(self.state_buffer) == 0:
            return
            
        # 将缓冲区数据转换为张量 - 确保是二维张量 [N, state_dim]
        state_reps = torch.tensor(np.array(self.state_buffer), device=self.device, dtype=torch.float32)
        gradients = torch.tensor(np.array(self.grad_buffer), device=self.device, dtype=torch.float32)
        
        # 计算注意力权重
        with torch.no_grad():
            # 使用最新状态作为查询
            latest_rep = state_reps[-1].unsqueeze(0)  # [1, state_dim]
            
            # 计算查询和键 - 确保输出是二维
            queries = self.query_net(latest_rep)  # [1, 64]
            keys = self.key_net(state_reps)        # [N, 64]
            
            # 计算注意力分数 - 确保是二维矩阵乘法
            attn_scores = torch.mm(queries, keys.t()) / torch.sqrt(torch.tensor(64.0, device=self.device))
            attn_weights = F.softmax(attn_scores, dim=-1)  # [1, N]
            
            # 加权平均历史梯度
            weighted_grad = torch.mm(attn_weights, gradients)  # [1, action_dim]梯度裁剪：对当前梯度也做上下界限制，保证数值稳定。
            
            # 指数移动平均更新全局经验向量
            beta = min(0.9, 1.0 - 1.0/(self.n_updates+1))
            self.experience.data = beta * self.experience + (1 - beta) * weighted_grad
        
    def get_direction(self, state_rep, current_grad, current_step=None):
        """获取融合方向，包含干预控制"""
        with torch.no_grad():
            # 1. 计算当前干预强度
            if current_step is None:
                intervention_factor = self.max_intervention
            else:
                # 线性增加干预强度
                progress = min(1.0, current_step / self.intervention_steps)
                # print(progress)
                intervention_factor = self.min_intervention + progress * (self.max_intervention - self.min_intervention)
        
            # 2. 当前梯度约束
            if self.grad_clip is not None:
                current_grad = torch.clamp(current_grad, -self.grad_clip, self.grad_clip)
            
            # 3. 计算余弦相似度
            exp_norm = torch.norm(self.experience, dim=1, keepdim=True)
            grad_norm = torch.norm(current_grad, dim=1, keepdim=True)
            dot_product = torch.sum(self.experience * current_grad, dim=1, keepdim=True)
            similarity = dot_product / (exp_norm * grad_norm + 1e-6)
            
            # 4. 自适应融合系数
            beta = torch.sigmoid(similarity * 5).clamp(0.1, 0.9)
            
            # 5. 融合当前梯度和经验向量
            fused_direction = beta * self.experience + (1 - beta) * current_grad
            
            # 6. 归一化处理
            norm = torch.norm(fused_direction, dim=1, keepdim=True)
            normalized_direction = fused_direction / (norm + 1e-6)
            
            return intervention_factor * normalized_direction
    
    def forward(self, state_rep, current_grad, current_step=None):
        return self.get_direction(state_rep, current_grad, current_step)

    def intervention_level(self, current_step=None):
        if current_step is None:
            return self.max_intervention
        progress = min(1.0, current_step / self.intervention_steps)
        return self.min_intervention + progress * (self.max_intervention - self.min_intervention)
    


class OTExperienceBank:
    """最优传输经验银行（修复维度匹配问题）"""
    def __init__(self, state_dim, action_dim, device, 
                 capacity=1000, 
                 sinkhorn_reg=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.capacity = capacity
        self.sinkhorn_reg = sinkhorn_reg
        
        # 经验存储
        self.states = deque(maxlen=capacity)
        self.means = deque(maxlen=capacity)
        self.stds = deque(maxlen=capacity)

    def update(self, state, mean, std):
        """更新经验银行"""
        # 确保使用float32
        state_np = state.detach().cpu().numpy().flatten().astype(np.float32)
        
        # 确保均值和标准差是1D数组
        mean_np = mean.detach().cpu().numpy().squeeze().astype(np.float32)
        std_np = std.detach().cpu().numpy().squeeze().astype(np.float32)
        
        # 如果action_dim=1，确保数组形状正确
        if mean_np.ndim == 0:
            mean_np = np.array([mean_np])
        if std_np.ndim == 0:
            std_np = np.array([std_np])
        
        self.states.append(state_np)
        self.means.append(mean_np)
        self.stds.append(std_np)

    def get_ot_mapping(self, state, current_mean, current_std, iterations):
        """获取最优传输映射（修复维度问题）"""
        Beta = max(1 - 1/94000* iterations, 0.1)

        if len(self.states) == 0:
            return None
        # 确保使用float32
        state_np = state.detach().cpu().numpy().flatten().astype(np.float32)
        
        # 计算状态相似度
        states = np.array(self.states, dtype=np.float32)
        dists = cdist([state_np], states)[0]
   
        # 选择最相似的k个 (修复边界问题)
        k = min(5, len(dists))
        if k <= 1:  # 当银行中只有1个经验时
            topk_idxs = [0] if k == 1 else []
        else:
            topk_idxs = np.argpartition(dists, min(k-1, len(dists)-1))[:k]  # 确保kth在边界内
        
        if len(topk_idxs) == 0:
            return None
            
        # 获取历史分布
        hist_means = [self.means[i] for i in topk_idxs]
        hist_stds = [self.stds[i] for i in topk_idxs]
        
        # 转换为float32张量
        hist_means_t = torch.tensor(np.array(hist_means), 
                                  dtype=torch.float32, 
                                  device=self.device)
        hist_stds_t = torch.tensor(np.array(hist_stds), 
                                 dtype=torch.float32, 
                                 device=self.device)
        # print(dists)
        # 计算加权平均历史分布
        weights = 1.0 / (torch.tensor(dists[topk_idxs], 
                                    dtype=torch.float32, 
                                    device=self.device) + 1e-8)
        weights = weights / weights.sum()
        # 确保维度匹配
        if hist_means_t.dim() == 1:
            hist_means_t = hist_means_t.unsqueeze(1)
        if weights.dim() == 1:
            weights = weights.view(-1, 1)
        # print(hist_means_t.shape)
        # print(hist_stds_t.shape)
        if iterations > 10000:
            weights = weights * Beta
        if hist_means_t.shape[0] < 5:
            # 先提取 top n 的 mean 和 std
            hist_mean = torch.sum(hist_means_t * weights[:hist_means_t.shape[0]].unsqueeze(-1), dim=0)
            hist_std = torch.sum(hist_stds_t * weights[:hist_means_t.shape[0]], dim=0)
        else:
            hist_mean = torch.sum(hist_means_t * weights.unsqueeze(-1), dim=0)
            hist_std = torch.sum(hist_stds_t * weights, dim=0)
        current_mean = current_mean.to(torch.float32)
        current_std = current_std.to(torch.float32)

        # 创建新分布
        current_dist = torch.distributions.Normal(current_mean, current_std)
        hist_dist = torch.distributions.Normal(hist_mean, hist_std)
        
        # 计算最优传输
        return self._compute_ot_mapping(current_dist, hist_dist)

    def _compute_ot_mapping(self, source_dist, target_dist, n_samples=50):
        """计算最优传输映射（修复维度问题）"""
        # 从两个分布采样并确保float32
        source_samples = source_dist.rsample((n_samples,)).to(torch.float32)
        target_samples = target_dist.rsample((n_samples,)).to(torch.float32)
        
        # 重塑样本为二维矩阵 [n_samples, d]
        d = source_samples.shape[-1]
        source_samples = source_samples.reshape(n_samples, -1)
        target_samples = target_samples.reshape(n_samples, -1)
        
        # 计算成本矩阵
        C = torch.cdist(source_samples, target_samples, p=2)  # [n_samples, n_samples]
        
        # Sinkhorn算法
        K = torch.exp(-C / self.sinkhorn_reg)  # [n_samples, n_samples]
        
        # 确保向量是列向量
        u = torch.ones(n_samples, 1, device=self.device, dtype=torch.float32) / n_samples  # [n_samples, 1]
        v = torch.ones(n_samples, 1, device=self.device, dtype=torch.float32) / n_samples  # [n_samples, 1]
        
        # 迭代优化（修复矩阵乘法维度）
        for _ in range(50):
            u = 1.0 / (torch.mm(K, v) + 1e-8)  # [n_samples, 1]
            v = 1.0 / (torch.mm(K.t(), u) + 1e-8)  # [n_samples, 1]
        
        # 计算传输矩阵 (使用广播)
        P = u * K * v.t()  # [n_samples, n_samples]
        
        return P
