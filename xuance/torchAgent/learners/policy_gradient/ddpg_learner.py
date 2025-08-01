# ddpg_learner.py
from xuance.torchAgent.learners import *
from xuance.gradient_bank import GradientBank

class DDPG_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizers: Sequence[torch.optim.Optimizer],
                 schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 use_grad_bank: bool = False,
                 grad_bank_kwargs: dict = None): 
        self.tau = tau
        self.gamma = gamma
        self.use_grad_bank = use_grad_bank
        super(DDPG_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)
        
        if self.use_grad_bank:
            # 获取状态表示的维度
            state_dim = self.policy.actor_representation.output_shapes['state'][-1]
            action_dim = self.policy.action_dim
            
            # 默认梯度银行参数
            default_kwargs = {
                'state_dim': state_dim,
                'device': device,
                'capacity': 1000,
                'min_intervention': 0,
                'max_intervention': 0.01,
                'intervention_steps': 200000,
                'grad_clip': 1.0
            }
            
            # 合并用户提供的参数
            if grad_bank_kwargs:
                default_kwargs.update(grad_bank_kwargs)
            
            # 初始化梯度银行
            self.grad_bank = GradientBank(
                action_dim=action_dim,
                state_dim=state_dim,
                device=device,
                capacity=default_kwargs['capacity'],
                min_intervention=default_kwargs['min_intervention'],
                max_intervention=default_kwargs['max_intervention'],
                intervention_steps=default_kwargs['intervention_steps'],
                grad_clip=default_kwargs['grad_clip']
            )

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)
        obs_batch = torch.tensor(obs_batch, device=self.device)
        next_batch = torch.tensor(next_batch, device=self.device)
        
        # Critic update - 与原始DDPG完全一致
        action_q = self.policy.Qaction(obs_batch, act_batch).reshape([-1])
        next_q = self.policy.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + (1 - ter_batch) * self.gamma * next_q
        q_loss = F.mse_loss(action_q, target_q.detach())
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()
        
        # Actor update - 根据配置选择不同路径
        if self.use_grad_bank:
            # 使用梯度经验存储机制
            # 1. 获取actor表示
            outputs_actor = self.policy.actor_representation(obs_batch)
            rep_actor = outputs_actor['state']
            
            # 2. 生成动作
            a_det = self.policy.actor(rep_actor)
            
            # 3. 计算当前动作的Q值
            policy_q = self.policy.critic(rep_actor, a_det).mean()
            
            # 4. 计算梯度
            grad = torch.autograd.grad(policy_q, a_det, retain_graph=True)[0]
            current_grad = grad.detach()
        
            # 5. 获取融合方向（使用当前迭代步数）
            fused_direction = self.grad_bank.get_direction(
                state_rep=rep_actor.detach(), 
                current_grad=current_grad,
                current_step=self.iterations
            )
            
            #6. 应用定向扰动
            delta = 0.1 * fused_direction
            rad_max     = (2 * self.grad_bank.max_intervention) ** 0.5 * 0.1
            delta_norm  = delta.norm(dim=1, keepdim=True)  # [B, 1]
            delta = torch.where(
                delta_norm > rad_max,
                delta * (rad_max / (delta_norm + 1e-8)),
                delta
            )
            a_blend = (a_det + delta).clamp(-1, 1)
                
            
            # 7. 计算混合动作的Q值
            policy_q_blend = self.policy.critic(rep_actor, a_blend).mean()
            
            # 8. 策略优化（使用混合动作的Q值）
            p_loss = -policy_q_blend
        else:
            # 与原始DDPG完全一致
            policy_q = self.policy.Qpolicy(obs_batch)
            p_loss = -policy_q.mean()
        
        # 策略优化
        self.optimizer[0].zero_grad()
        p_loss.backward()
        self.optimizer[0].step()
        
        # 如果启用梯度经验存储，更新经验
        if self.use_grad_bank:
            # 1. 获取actor表示（分离计算图）
            with torch.no_grad():
                outputs_actor = self.policy.actor_representation(obs_batch)
                rep_actor = outputs_actor['state']
            
            # 2. 生成动作并启用梯度
            a_det = self.policy.actor(rep_actor)
            a_det.requires_grad_(True)
            
            # 3. 计算Q值
            q_val = self.policy.critic(rep_actor, a_det).mean()
            
            # 4. 计算梯度
            new_grad = torch.autograd.grad(q_val, a_det, retain_graph=False)[0]
            
            # 5. 更新经验（传入状态表示和梯度）
            mean_state = rep_actor.mean(dim=0, keepdim=True).detach()
            mean_grad  = new_grad .mean(dim=0, keepdim=True).detach()
            self.grad_bank.update(mean_state, mean_grad)

        if self.scheduler is not None:
            self.scheduler[0].step()
            self.scheduler[1].step()

        # 策略软更新 - 与原始DDPG一致
        self.policy.soft_update(self.tau)

        # 日志记录
        actor_lr = self.optimizer[0].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer[1].state_dict()['param_groups'][0]['lr']

        info = {
            "Qloss": q_loss.item(),
            "Ploss": p_loss.item(),
            "Qvalue": action_q.mean().item(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr
        }
        
        if self.use_grad_bank:
            info["ExpNorm"] = torch.norm(self.grad_bank.experience).item()
            info["Intervention"] = self.grad_bank.intervention_level(self.iterations)
            # 添加缓冲区信息
            info["BufferSize"] = len(self.grad_bank.state_buffer)

        return info