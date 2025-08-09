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
    


# class OTExperienceBank:
#     """最优传输经验银行（修复维度匹配问题）"""
#     def __init__(self, state_dim, action_dim, device, 
#                  capacity=1000, 
#                  sinkhorn_reg=0.1):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.device = device
#         self.capacity = capacity
#         self.sinkhorn_reg = sinkhorn_reg
        
#         # 经验存储
#         self.states = deque(maxlen=capacity)
#         self.means = deque(maxlen=capacity)
#         self.stds = deque(maxlen=capacity)

#     def update(self, state, mean, std):
#         """更新经验银行"""
#         # 确保使用float32
#         state_np = state.detach().cpu().numpy().flatten().astype(np.float32)
        
#         # 确保均值和标准差是1D数组
#         mean_np = mean.detach().cpu().numpy().squeeze().astype(np.float32)
#         std_np = std.detach().cpu().numpy().squeeze().astype(np.float32)
        
#         # 如果action_dim=1，确保数组形状正确
#         if mean_np.ndim == 0:
#             mean_np = np.array([mean_np])
#         if std_np.ndim == 0:
#             std_np = np.array([std_np])
        
#         self.states.append(state_np)
#         self.means.append(mean_np)
#         self.stds.append(std_np)

#     def get_ot_mapping(self, state, current_mean, current_std, iterations):
#         """获取最优传输映射（修复维度问题）"""
#         Beta = max(1 - 1/94000* iterations, 0.1)

#         if len(self.states) == 0:
#             return None
#         # 确保使用float32
#         state_np = state.detach().cpu().numpy().flatten().astype(np.float32)
        
#         # 计算状态相似度
#         states = np.array(self.states, dtype=np.float32)
#         dists = cdist([state_np], states)[0]
   
#         # 选择最相似的k个 (修复边界问题)
#         k = min(5, len(dists))
#         if k <= 1:  # 当银行中只有1个经验时
#             topk_idxs = [0] if k == 1 else []
#         else:
#             topk_idxs = np.argpartition(dists, min(k-1, len(dists)-1))[:k]  # 确保kth在边界内
        
#         if len(topk_idxs) == 0:
#             return None
            
#         # 获取历史分布
#         hist_means = [self.means[i] for i in topk_idxs]
#         hist_stds = [self.stds[i] for i in topk_idxs]
        
#         # 转换为float32张量
#         hist_means_t = torch.tensor(np.array(hist_means), 
#                                   dtype=torch.float32, 
#                                   device=self.device)
#         hist_stds_t = torch.tensor(np.array(hist_stds), 
#                                  dtype=torch.float32, 
#                                  device=self.device)
#         # print(dists)
#         # 计算加权平均历史分布
#         weights = 1.0 / (torch.tensor(dists[topk_idxs], 
#                                     dtype=torch.float32, 
#                                     device=self.device) + 1e-8)
#         weights = weights / weights.sum()
#         # 确保维度匹配
#         if hist_means_t.dim() == 1:
#             hist_means_t = hist_means_t.unsqueeze(1)
#         if weights.dim() == 1:
#             weights = weights.view(-1, 1)
#         # print(hist_means_t.shape)
#         # print(hist_stds_t.shape)
#         if iterations > 10000:
#             weights = weights * Beta
#         if hist_means_t.shape[0] < 5:
#             # 先提取 top n 的 mean 和 std
#             hist_mean = torch.sum(hist_means_t * weights[:hist_means_t.shape[0]].unsqueeze(-1), dim=0)
#             hist_std = torch.sum(hist_stds_t * weights[:hist_means_t.shape[0]], dim=0)
#         else:
#             hist_mean = torch.sum(hist_means_t * weights.unsqueeze(-1), dim=0)
#             hist_std = torch.sum(hist_stds_t * weights, dim=0)
#         current_mean = current_mean.to(torch.float32)
#         current_std = current_std.to(torch.float32)

#         # 创建新分布
#         current_dist = torch.distributions.Normal(current_mean, current_std)
#         hist_dist = torch.distributions.Normal(hist_mean, hist_std)
        
#         # 计算最优传输
#         return self._compute_ot_mapping(current_dist, hist_dist)

#     def _compute_ot_mapping(self, source_dist, target_dist, n_samples=50):
#         """计算最优传输映射（修复维度问题）"""
#         # 从两个分布采样并确保float32
#         source_samples = source_dist.rsample((n_samples,)).to(torch.float32)
#         target_samples = target_dist.rsample((n_samples,)).to(torch.float32)
        
#         # 重塑样本为二维矩阵 [n_samples, d]
#         d = source_samples.shape[-1]
#         source_samples = source_samples.reshape(n_samples, -1)
#         target_samples = target_samples.reshape(n_samples, -1)
        
#         # 计算成本矩阵
#         C = torch.cdist(source_samples, target_samples, p=2)  # [n_samples, n_samples]
        
#         # Sinkhorn算法
#         K = torch.exp(-C / self.sinkhorn_reg)  # [n_samples, n_samples]
        
#         # 确保向量是列向量
#         u = torch.ones(n_samples, 1, device=self.device, dtype=torch.float32) / n_samples  # [n_samples, 1]
#         v = torch.ones(n_samples, 1, device=self.device, dtype=torch.float32) / n_samples  # [n_samples, 1]
        
#         # 迭代优化（修复矩阵乘法维度）
#         for _ in range(50):
#             u = 1.0 / (torch.mm(K, v) + 1e-8)  # [n_samples, 1]
#             v = 1.0 / (torch.mm(K.t(), u) + 1e-8)  # [n_samples, 1]
        
#         # 计算传输矩阵 (使用广播)
#         P = u * K * v.t()  # [n_samples, n_samples]
        
#         return P

# # 1. 计算状态相似度（使用马氏距离）
#     state_np = state.detach().cpu().numpy().flatten().astype(np.float32)
#     states = np.array(self.states, dtype=np.float32)
    
#     # 计算状态协方差矩阵
#     cov_matrix = np.cov(states.T) + 1e-6 * np.eye(states.shape[1])
#     inv_cov = np.linalg.inv(cov_matrix)
    
#     # 计算马氏距离
#     diff = states - state_np
#     dists = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    
#     # 2. 动态选择k值
#     k = min(10, max(3, int(len(self.states)**0.5)))  # 基于经验库大小的平方根
    
#     # 3. 选择top-k索引
#     topk_idxs = np.argsort(dists)[:k] if len(dists) > 0 else []
    
#     if len(topk_idxs) == 0:
#         return None
    
#     # 4. 基于最优传输理论的权重计算
#     topk_dists = dists[topk_idxs]
    
#     # 计算权重基值（使用指数衰减）
#     base_weights = np.exp(-topk_dists / np.median(topk_dists))
    
#     # 5. 基于迭代次数的调整
#     # 训练初期更依赖历史，后期更依赖当前策略
#     history_weight = max(0.3, 1.0 - min(0.7, iterations/50000))
    
#     # 6. 基于KL散度的质量评估
#     kl_weights = []
#     for i in topk_idxs:
#         hist_dist = torch.distributions.Normal(
#             torch.tensor(self.means[i], device=self.device),
#             torch.tensor(self.stds[i], device=self.device)
#         )
#         kl_div = torch.distributions.kl_divergence(hist_dist, current_dist).mean().item()
#         kl_weights.append(np.exp(-kl_div))
    
#     # 7. 组合权重
#     combined_weights = base_weights * np.array(kl_weights) * history_weight
#     weights = torch.tensor(combined_weights / combined_weights.sum(), 
#                           dtype=torch.float32, device=self.device)

class OTExperienceBank:
    """统一的最优传输经验银行，支持连续和离散动作空间"""
    def __init__(self, state_dim, action_dim, dist_type, device, 
                 capacity=1000, sinkhorn_reg=0.1):
        """
        dist_type: 'gaussian' 或 'categorical'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dist_type = dist_type
        self.device = device
        self.capacity = capacity
        self.sinkhorn_reg = sinkhorn_reg
        
        # 经验存储
        self.states = deque(maxlen=capacity)
        self.dist_params = deque(maxlen=capacity)  # 存储分布参数
        
        # 根据分布类型选择存储方式
        if dist_type == 'gaussian':
            self._update = self._update_gaussian
            self._get_ot_mapping = self._get_ot_mapping_gaussian
            self._compute_ot_mapping = self._compute_ot_mapping_gaussian
            self._apply_ot_mapping = self._apply_ot_mapping_gaussian
        elif dist_type == 'categorical':
            self._update = self._update_categorical
            self._get_ot_mapping = self._get_ot_mapping_categorical
            self._compute_ot_mapping = self._compute_ot_mapping_categorical
            self._apply_ot_mapping = self._apply_ot_mapping_categorical
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    def update(self, state, dist_params):
        """更新经验银行"""
        self._update(state, dist_params)
    
    def get_ot_mapping(self, state, current_params, iterations):
        """获取最优传输映射"""
        return self._get_ot_mapping(state, current_params, iterations)
    
    def apply_ot_mapping(self, current_params, transport_map):
        """应用OT映射到当前分布参数"""
        return self._apply_ot_mapping(current_params, transport_map)
    
    # ===== 高斯分布专用方法 =====
    def _update_gaussian(self, state, dist_params):
        """更新高斯分布的经验银行"""
        state_np = state.detach().cpu().numpy().flatten().astype(np.float32)
        mean, std = dist_params
        
        # 确保数组形状正确
        mean_np = mean.detach().cpu().numpy().squeeze().astype(np.float32)
        std_np = std.detach().cpu().numpy().squeeze().astype(np.float32)
        
        if mean_np.ndim == 0:
            mean_np = np.array([mean_np])
        if std_np.ndim == 0:
            std_np = np.array([std_np])
        
        self.states.append(state_np)
        self.dist_params.append((mean_np, std_np))
    
    def _get_ot_mapping_gaussian(self, state, current_params, iterations):
        """获取高斯分布的最优传输映射"""
        if len(self.states) == 0:
            return None
            
        state_np = state.detach().cpu().numpy().flatten().astype(np.float32)
        states = np.array(self.states, dtype=np.float32)
        
        # 计算混合相似度
        cos_sim = np.dot(states, state_np) / (np.linalg.norm(states, axis=1) * np.linalg.norm(state_np) + 1e-8)
        max_norms = np.maximum(np.linalg.norm(states, axis=1), np.linalg.norm(state_np))
        euc_dist = np.linalg.norm(states - state_np, axis=1) / (max_norms + 1e-8)
        similarity = 0.7 * cos_sim + 0.3 * (1 - euc_dist)
        dists = 1 - similarity
        
        # 动态选择k值
        
        k = min(10, max(3, int(len(self.states)**0.5)))
        
        topk_idxs = np.argsort(dists)[:k] if len(dists) > 0 else []
        if len(topk_idxs) == 0:
            return None
            
        # 获取历史分布参数
        hist_params = [self.dist_params[i] for i in topk_idxs]
        hist_means = [p[0] for p in hist_params]
        hist_stds = [p[1] for p in hist_params]
        
        # 转换为张量
        hist_means_t = torch.tensor(np.array(hist_means), dtype=torch.float32, device=self.device)
        hist_stds_t = torch.tensor(np.array(hist_stds), dtype=torch.float32, device=self.device)
        
        # 计算权重
        topk_dists = dists[topk_idxs]
        base_weights = np.exp(-topk_dists / (np.median(topk_dists) + 1e-8))
        weights = torch.tensor(base_weights / base_weights.sum(), 
                              dtype=torch.float32, device=self.device)
        
        # 计算加权平均历史分布
        if hist_means_t.dim() == 1:
            hist_means_t = hist_means_t.unsqueeze(1)
        weights = weights.view(-1, 1)
        
        if hist_means_t.shape[0] < 10:
            # 先提取 top n 的 mean 和 std
            hist_mean = torch.sum(hist_means_t * weights[:hist_means_t.shape[0]].unsqueeze(-1), dim=0)
            hist_std = torch.sum(hist_stds_t * weights[:hist_means_t.shape[0]], dim=0)
        else:
            hist_mean = torch.sum(hist_means_t * weights.unsqueeze(-1), dim=0)
            hist_std = torch.sum(hist_stds_t * weights, dim=0)
        
        # 创建分布
        current_mean, current_std = current_params
        current_dist = torch.distributions.Normal(current_mean, current_std)
        hist_dist = torch.distributions.Normal(hist_mean, hist_std)
        
        # 计算最优传输
        return self._compute_ot_mapping(current_dist, hist_dist)
    
    def _compute_ot_mapping_gaussian(self, source_dist, target_dist, n_samples=50):
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
        # """计算高斯分布的最优传输映射"""
        # # 从两个分布采样
        # source_samples = source_dist.rsample((n_samples,)).to(torch.float32)
        # target_samples = target_dist.rsample((n_samples,)).to(torch.float32)
        
        # # 计算源和目标样本的均值
        # source_mean = source_samples.mean(dim=0)
        # target_mean = target_samples.mean(dim=0)
        
        # # 计算协方差矩阵
        # cov_source = torch.cov(source_samples.T)
        # cov_target = torch.cov(target_samples.T)
        
        # # 添加正则化确保正定
        # cov_source += 1e-4 * torch.eye(cov_source.size(0), device=self.device)
        # cov_target += 1e-4 * torch.eye(cov_target.size(0), device=self.device)
        
        # # 计算最优传输映射矩阵
        # cov_source_sqrt = torch.linalg.cholesky(cov_source)
        # cov_target_sqrt = torch.linalg.cholesky(cov_target)
        
        # # 计算映射矩阵: T = cov_target^{1/2} @ cov_source^{-1/2}
        # T = cov_target_sqrt @ torch.linalg.inv(cov_source_sqrt)
        
        # return T
    
    def _apply_ot_mapping_gaussian(self, current_params, transport_map):
        """应用OT映射到高斯分布参数"""
        n_samples = 50
        mean, std = current_params
        # 创建分布
        dist = torch.distributions.Normal(mean, std)
    
        # 采样并重塑为二维
        samples = dist.rsample((n_samples,))  # [n_samples, ...]
        original_shape = samples.shape[1:]
        samples = samples.reshape(n_samples, -1)  # [n_samples, d]

        # 应用传输映射: aligned_samples = transport_map^T @ samples
        aligned_samples = torch.einsum('ij,jk -> ik', transport_map.t(), samples)
    
        # 计算新分布参数
        aligned_mean = aligned_samples.mean(dim=0)  # [d]
        aligned_std = aligned_samples.std(dim=0).clamp(min=1e-3, max=2.0)   # [d]
    
        # 重塑回原始形状
        aligned_mean = aligned_mean.reshape(original_shape)
        aligned_std = aligned_std.reshape(original_shape)
    
        return aligned_mean, aligned_std
        # aligned_mean = torch.mv(transport_map, mean)
        
        # # 变换协方差矩阵: T @ diag(std^2) @ T^T
        # std_sq = std ** 2
        # aligned_cov = transport_map @ torch.diag(std_sq) @ transport_map.t()
        # aligned_std = torch.sqrt(torch.diag(aligned_cov)).clamp(min=1e-3)
        
        # return aligned_mean, aligned_std
    
    # ===== 分类分布专用方法 =====
    def _update_categorical(self, state, dist_params):
        """更新分类分布的经验银行"""
        state_np = state.detach().cpu().numpy().flatten().astype(np.float32)
        probs = dist_params
        
        # 确保概率分布有效
        probs_np = probs.detach().cpu().numpy().astype(np.float32)
        probs_np = np.clip(probs_np, 1e-5, 1.0)
        probs_np /= probs_np.sum()
        
        self.states.append(state_np)
        self.dist_params.append(probs_np)
    
    def _get_ot_mapping_categorical(self, state, current_params, iterations):
        """获取分类分布的最优传输映射"""
        if len(self.states) == 0:
            return None
            
        state_np = state.detach().cpu().numpy().flatten().astype(np.float32)
        states = np.array(self.states, dtype=np.float32)
        
        # 计算混合相似度
        cos_sim = np.dot(states, state_np) / (np.linalg.norm(states, axis=1) * np.linalg.norm(state_np) + 1e-8)
        max_norms = np.maximum(np.linalg.norm(states, axis=1), np.linalg.norm(state_np))
        euc_dist = np.linalg.norm(states - state_np, axis=1) / (max_norms + 1e-8)
        similarity = 0.7 * cos_sim + 0.3 * (1 - euc_dist)
        dists = 1 - similarity
        
        # 动态选择k值
        k = min(10, max(3, int(len(self.states)**0.5)))
        topk_idxs = np.argsort(dists)[:k] if len(dists) > 0 else []
        
        if len(topk_idxs) == 0:
            return None
            
        # 获取历史概率分布
        hist_probs = [self.dist_params[i] for i in topk_idxs]
        hist_probs_t = torch.tensor(np.array(hist_probs), dtype=torch.float32, device=self.device)
        # print(hist_probs_t)
        # 计算权重
        topk_dists = dists[topk_idxs]
        base_weights = np.exp(-topk_dists / (np.median(topk_dists) + 1e-8))
        weights = torch.tensor(base_weights / base_weights.sum(), 
                              dtype=torch.float32, device=self.device)
        # print(weights)
        # 计算加权平均历史分布
        weights = weights.view(-1, 1)
        hist_probs_avg = torch.sum(hist_probs_t * weights, dim=0)
        
        # 确保概率有效
        current_probs = torch.softmax(current_params, dim=-1)
        hist_probs_avg = torch.softmax(hist_probs_avg, dim=-1)
        
        # 计算最优传输映射
        return self._compute_ot_mapping(current_probs, hist_probs_avg)
    
    def _compute_ot_mapping_categorical(self, source_probs, target_probs):
        """计算分类分布的最优传输映射（数值稳定版）"""
        eps = 1e-8
        # 成本矩阵：使用概率向量的JS散度
        C = torch.zeros((self.action_dim, self.action_dim), device=self.device)
        
        def kl_divergence(p, q):
            """稳定 KL(P || Q)"""
            p = p + eps
            q = q + eps
            return torch.sum(p * torch.log(p / q))
        
        # 计算所有动作对之间的成本
        for i in range(self.action_dim):
            for j in range(self.action_dim):
                # 创建单位向量表示特定动作
                p_i = torch.zeros(self.action_dim, device=self.device)
                p_i[i] = 1.0
                p_j = torch.zeros(self.action_dim, device=self.device)
                p_j[j] = 1.0
                
                # 计算JS散度作为成本
                m = 0.5 * (p_i + p_j)
                js_div = 0.5 * kl_divergence(p_i, m) + 0.5 * kl_divergence(p_j, m)
                C[i, j] = js_div

        # print(C)
        # Sinkhorn算法
        K = torch.exp(-C / self.sinkhorn_reg)
        u = torch.ones(self.action_dim, 1, device=self.device) / self.action_dim
        v = torch.ones(self.action_dim, 1, device=self.device) / self.action_dim
        
        for _ in range(50):
            u = source_probs.view(-1, 1) / (K @ v + eps)
            v = target_probs.view(-1, 1) / (K.t() @ u + eps)
        
        # 传输矩阵
        P = u * K * v.t()
        P = P / P.sum()  # 归一化
        
        return P
    
    def _apply_ot_mapping_categorical(self, current_params, transport_map):
        """应用OT映射到分类分布参数"""
        # 应用传输映射: aligned_probs = P^T @ probs
        probs = torch.softmax(current_params, dim=-1)
        
        aligned_probs = torch.mv(transport_map.t(), probs)
    
        # 确保概率有效
        aligned_probs = torch.clamp(aligned_probs, min=1e-5)
        aligned_probs /= aligned_probs.sum()
        
        # 返回logits格式
        return torch.log(aligned_probs + 1e-8)
