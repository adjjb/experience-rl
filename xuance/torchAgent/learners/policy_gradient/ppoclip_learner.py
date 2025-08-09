from xuance.torchAgent.learners import *
from xuance.gradient_bank import OTExperienceBank

class PPOCLIP_Learner(Learner):
    # def __init__(self,
    #              policy: nn.Module,
    #              optimizer: torch.optim.Optimizer,
    #              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    #              device: Optional[Union[int, str, torch.device]] = None,
    #              model_dir: str = "./",
    #              vf_coef: float = 0.25,
    #              ent_coef: float = 0.005,
    #              clip_range: float = 0.25,
    #              clip_grad_norm: float = 0.25,
    #              use_grad_clip: bool = True,
    #              use_grad_bank: bool = False,
    #              ot_bank_capacity: int = 1000,
    #              ot_update_freq: int = 15):
        
    #     super(PPOCLIP_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)

    #     self.vf_coef = vf_coef
    #     self.ent_coef = ent_coef
    #     self.clip_range = clip_range
    #     self.clip_grad_norm = clip_grad_norm
    #     self.use_grad_clip = use_grad_clip

        
    #     # 新增OT银行功能
    #     self.use_grad_bank = use_grad_bank
    #     self.ot_update_freq = ot_update_freq
        
    #     if self.use_grad_bank:
    #         state_dim = policy.representation.output_shapes['state'][0]
    #         action_dim = policy.action_dim
    #         self.ot_bank = OTExperienceBank(
    #             state_dim=state_dim,
    #             action_dim=action_dim,
    #             device=device,
    #             capacity=ot_bank_capacity
    #         )

    # def update(self, obs_batch, act_batch, ret_batch, value_batch, adv_batch, old_logp):
    #     self.iterations += 1
    #     # print(self.iterations)
    #     # 转换为张量并确保float32
    #     act_batch = torch.as_tensor(act_batch, device=self.device, dtype=torch.float32)
    #     ret_batch = torch.as_tensor(ret_batch, device=self.device, dtype=torch.float32)
    #     value_batch = torch.as_tensor(value_batch, device=self.device, dtype=torch.float32)
    #     adv_batch = torch.as_tensor(adv_batch, device=self.device, dtype=torch.float32)
    #     old_logp_batch = torch.as_tensor(old_logp, device=self.device, dtype=torch.float32)
    #     obs_batch = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)

    #     # 获取策略输出
    #     outputs, a_dist, v_pred = self.policy(obs_batch)
    #     # print("outputs:", outputs['state'].shape)
    #     # 获取分布参数并确保float32
    #     mean, std = a_dist.get_param()
    #     mean = mean.to(torch.float32)
    #     std = std.to(torch.float32)
        
    #     # OT分布对齐
    #     if self.use_grad_bank and (self.iterations % self.ot_update_freq == 0):
    #         # 使用批量平均状态并确保float32
    #         avg_state = outputs['state'].mean(dim=0, keepdim=True).to(torch.float32)
    #         # print(avg_state.shape)
    #         # 获取OT映射
    #         transport_map = self.ot_bank.get_ot_mapping(
    #             state=avg_state,
    #             current_mean=mean,
    #             current_std=std,
    #             iterations=self.iterations
    #         )
            
    #         if transport_map is not None:
    #             # 应用OT映射
    #             aligned_mean, aligned_std = self._apply_ot_mapping(mean, std, transport_map)
                
    #             # 创建新分布（确保float32）
    #             a_dist.set_param(aligned_mean.to(torch.float32), 
    #                            aligned_std.to(torch.float32))
            
    #         # 更新经验银行
    #         # if self.iterations % (10 * self.ot_update_freq) == 0:
    #         if self.iterations % (5 * self.ot_update_freq) == 0:
    #             self.ot_bank.update(
    #                 state=avg_state,
    #                 mean=mean,
    #                 std=std
    #             )
        
    #     # 使用分布计算对数概率
    #     log_prob = a_dist.log_prob(act_batch)
        
    #     # PPO损失计算
    #     ratio = (log_prob - old_logp_batch).exp().float()
    #     surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
    #     surrogate2 = adv_batch * ratio
    #     a_loss = -torch.minimum(surrogate1, surrogate2).mean()
        
    #     c_loss = F.mse_loss(v_pred, ret_batch)
    #     e_loss = a_dist.entropy().mean()
        
    #     total_loss = a_loss + self.vf_coef * c_loss - self.ent_coef * e_loss
        
    #     # 优化步骤
    #     self.optimizer.zero_grad()
    #     total_loss.backward()
    #     if self.use_grad_clip:
    #         torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
    #     self.optimizer.step()
        
    #     if self.scheduler is not None:
    #         self.scheduler.step()
        
    #     # 日志记录
    #     lr = self.optimizer.state_dict()['param_groups'][0]['lr']
    #     cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]
        
    #     info = {
    #         "actor-loss": a_loss.item(),
    #         "critic-loss": c_loss.item(),
    #         "entropy": e_loss.item(),
    #         "learning_rate": lr,
    #         "predict_value": v_pred.mean().item(),
    #         "clip_ratio": cr
    #     }
        
    #     if self.use_grad_bank and (self.iterations % self.ot_update_freq == 0):
    #         info["ot_update"] = 1
        
    #     return info
    

    # # def _apply_ot_mapping(self, mean, std, transport_map, n_samples=50):
    # #     """应用OT映射到分布（修复维度问题）"""
    # #     # 创建分布
    # #     dist = torch.distributions.Normal(mean, std)
    
    # #     # 采样并重塑为二维
    # #     samples = dist.rsample((n_samples,))  # [n_samples, ...]
    # #     original_shape = samples.shape[1:]
    # #     samples = samples.reshape(n_samples, -1)  # [n_samples, d]
    
    # #     # 应用传输映射: aligned_samples = transport_map^T @ samples
    # #     aligned_samples = torch.mm(transport_map.t(), samples)  # [n_samples, d]
    
    # #     # 计算新分布参数
    # #     aligned_mean = aligned_samples.mean(dim=0)  # [d]
    # #     aligned_std = aligned_samples.std(dim=0)    # [d]
    
    # #     # 确保标准差有效
    # #     aligned_std = torch.clamp(aligned_std, min=1e-4)
    
    # #     # 重塑回原始形状
    # #     aligned_mean = aligned_mean.reshape(original_shape)
    # #     aligned_std = aligned_std.reshape(original_shape)
    
    # #     return aligned_mean, aligned_std
    # def _apply_ot_mapping(self, mean, std, transport_map, n_samples=50):
    #     """应用OT映射到分布（修复维度问题）"""
    #     # 创建分布
    #     dist = torch.distributions.Normal(mean, std)
    
    #     # 采样并重塑为二维
    #     samples = dist.rsample((n_samples,))  # [n_samples, ...]
    #     original_shape = samples.shape[1:]
    #     samples = samples.reshape(n_samples, -1)  # [n_samples, d]

    #     # 应用传输映射: aligned_samples = transport_map^T @ samples
    #     aligned_samples = torch.einsum('ij,jk -> ik', transport_map.t(), samples)
    
    #     # 计算新分布参数
    #     aligned_mean = aligned_samples.mean(dim=0)  # [d]
    #     aligned_std = aligned_samples.std(dim=0).clamp(min=1e-3, max=2.0)   # [d]
    
    #     # 重塑回原始形状
    #     aligned_mean = aligned_mean.reshape(original_shape)
    #     aligned_std = aligned_std.reshape(original_shape)
    
    #     return aligned_mean, aligned_std

    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 clip_range: float = 0.25,
                 clip_grad_norm: float = 0.25,
                 use_grad_clip: bool = True,
                 use_grad_bank: bool = False,
                 ot_bank_capacity: int = 1000,
                 ot_update_freq: int = 15,
                 dist_type: str = "Categorical_AC"):
        super(PPOCLIP_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)

        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.clip_grad_norm = clip_grad_norm
        self.use_grad_clip = use_grad_clip
        self.use_grad_bank = use_grad_bank
        self.ot_update_freq = ot_update_freq
        self.last_kl = 0.0
        self.dist_type = dist_type
        self.is_discrete = True if self.dist_type == 'Categorical_AC' else False
        if self.use_grad_bank:
            state_dim = policy.representation.output_shapes['state'][0]
            action_dim = policy.action_dim
            dtype = 'categorical' if self.dist_type == 'Categorical_AC' else 'gaussian'

            self.ot_bank = OTExperienceBank(
                state_dim=state_dim,
                action_dim=action_dim,
                dist_type=dtype,
                device=device,
                capacity=ot_bank_capacity
            )

    def update(self, obs_batch, act_batch, ret_batch, value_batch, adv_batch, old_logp):
        self.iterations += 1
        # 转换为张量并确保正确类型
        act_batch = torch.as_tensor(act_batch, device=self.device, 
                                   dtype=torch.int64 if self.is_discrete else torch.float32)
        ret_batch = torch.as_tensor(ret_batch, device=self.device, dtype=torch.float32)
        value_batch = torch.as_tensor(value_batch, device=self.device, dtype=torch.float32)
        adv_batch = torch.as_tensor(adv_batch, device=self.device, dtype=torch.float32)
        old_logp_batch = torch.as_tensor(old_logp, device=self.device, dtype=torch.float32)
        obs_batch = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)

        # 获取策略输出
        outputs, a_dist, v_pred = self.policy(obs_batch)
        
        # 根据离散/连续类型获取分布参数
        if self.is_discrete:
            # 离散动作空间
            dist_params = a_dist.get_param()[0]  # 取第一个样本的logits
        else:
            # 连续动作空间
            mean, std = a_dist.get_param()
            dist_params = (mean, std)
        
        # OT分布对齐
        if self.use_grad_bank:
            # 使用优势最大的状态作为代表
            rep_idx = torch.argmax(torch.abs(adv_batch))
            rep_state = outputs['state'][rep_idx].unsqueeze(0)
            avg_state = outputs['state'].mean(dim=0, keepdim=True).to(torch.float32)
            # # 计算策略变化（KL散度）
            # with torch.no_grad():
            #     _, old_a_dist, _ = self.policy(obs_batch)
            #     kl_div = torch.distributions.kl_divergence(a_dist, old_a_dist).mean().item()
            #     self.last_kl = 0.9 * self.last_kl + 0.1 * kl_div
            
            # 动态更新条件
            # update_ot = (self.last_kl > 0.05) or (self.iterations % self.ot_update_freq == 0)
            # update_bank = (self.last_kl > 0.1) or (self.iterations % (self.ot_update_freq * 5) == 0)
            update_ot = (self.iterations % self.ot_update_freq == 0)
            update_bank = (self.iterations % (self.ot_update_freq * 5) == 0)
            if update_ot:
                transport_map = self.ot_bank.get_ot_mapping(
                    state=avg_state,
                    current_params=dist_params,
                    iterations=self.iterations
                )
                
                if transport_map is not None:
                    aligned_params = self.ot_bank.apply_ot_mapping(dist_params, transport_map)
                    
                    # 应用对齐后的参数
                    if self.is_discrete:
                        # 离散：对齐的是logits
                        a_dist.set_param(logits=aligned_params.unsqueeze(0).expand_as(a_dist.logits))
                    else:
                        # 连续：对齐的是(mean, std)
                        aligned_mean, aligned_std = aligned_params
                        a_dist.set_param(aligned_mean, aligned_std)
            
            if update_bank:
                # 使用代表性状态更新经验银行
                if self.is_discrete:
                    # 离散：存储logits
                    rep_params = a_dist.logits[rep_idx].detach()
                else:
                    # 连续：存储(mean, std)
                    # rep_mean, rep_std = a_dist.get_param()
                    # rep_mean = rep_mean[rep_idx].detach()
                    # rep_std = rep_std[rep_idx].detach()
                    # rep_params = (rep_mean, rep_std)
                    rep_params = dist_params

                
                self.ot_bank.update(
                    state=avg_state,
                    dist_params=rep_params
                )
        
        # PPO损失计算（通用部分）
        log_prob = a_dist.log_prob(act_batch)
        ratio = (log_prob - old_logp_batch).exp().float()
        surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -torch.minimum(surrogate1, surrogate2).mean()
        
        c_loss = F.mse_loss(v_pred, ret_batch)
        e_loss = a_dist.entropy().mean()
        total_loss = a_loss + self.vf_coef * c_loss - self.ent_coef * e_loss
        
        # 优化步骤
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        # 日志记录
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        cr = ((ratio < 1 - self.clip_range) | (ratio > 1 + self.clip_range)).float().mean()
        
        info = {
            "actor-loss": a_loss.item(),
            "critic-loss": c_loss.item(),
            "entropy": e_loss.item(),
            "learning_rate": lr,
            "predict_value": v_pred.mean().item(),
            "clip_ratio": cr.item()
            # "kl_divergence": self.last_kl
        }
        
        return info
