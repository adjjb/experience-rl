# td3_learner.py
from xuance.torchAgent.learners import *
from xuance.gradient_bank import GradientBank

class TD3_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizers: Sequence[torch.optim.Optimizer],
                 schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 delay: int = 3,
                 use_grad_bank: bool = False):
        self.tau = tau
        self.gamma = gamma
        self.delay = delay
        self.use_grad_bank = use_grad_bank
        super(TD3_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)
        
        if self.use_grad_bank:
            action_dim = self.policy.action_dim
            # 获取状态表示的维度
            state_dim = self.policy.actor_representation.output_shapes['state'][0]
            self.grad_bank = GradientBank(action_dim, state_dim, device)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)
        obs_batch = torch.tensor(obs_batch, device=self.device)
        next_batch = torch.tensor(next_batch, device=self.device)

        # Critic更新
        action_q_A, action_q_B = self.policy.Qaction(obs_batch, act_batch)
        action_q_A = action_q_A.reshape([-1])
        action_q_B = action_q_B.reshape([-1])
        next_q = self.policy.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + self.gamma * (1 - ter_batch) * next_q
        q_loss = F.mse_loss(action_q_A, target_q.detach()) + F.mse_loss(action_q_B, target_q.detach())
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()
        if self.scheduler is not None:
            self.scheduler[1].step()

        # Actor更新
        if self.iterations % self.delay == 0:
            if self.use_grad_bank:
                outputs_actor = self.policy.actor_representation(obs_batch)
                rep_actor = outputs_actor['state']
                a_det = self.policy.actor(rep_actor)
                
                outputs_critic_A = self.policy.critic_A_representation(obs_batch)
                rep_critic_A = outputs_critic_A['state']
                outputs_critic_B = self.policy.critic_B_representation(obs_batch)
                rep_critic_B = outputs_critic_B['state']
                
                q_A = self.policy.critic_A(rep_critic_A, a_det)
                q_B = self.policy.critic_B(rep_critic_B, a_det)
                policy_q = (q_A + q_B).mean() / 2.0
                
                grad = torch.autograd.grad(policy_q, a_det, retain_graph=True)[0]
                current_grad = grad.detach()
                
                # 使用批次平均状态表示和梯度
                mean_state_rep = rep_actor.mean(dim=0, keepdim=True)
                mean_grad = current_grad.mean(dim=0, keepdim=True)

                # print("Iteration:" + str(self.iterations))
                # 获取融合方向
                fused_direction = self.grad_bank.get_direction(
                    mean_state_rep, 
                    mean_grad,
                    current_step=self.iterations
                )
                
                # 应用定向扰动
                delta = 0.1 * fused_direction
                rad_max     = (2 * self.grad_bank.max_intervention) ** 0.5 * 0.1
                delta_norm  = delta.norm(dim=1, keepdim=True)  # [B, 1]
                delta = torch.where(
                    delta_norm > rad_max,
                    delta * (rad_max / (delta_norm + 1e-8)),
                    delta
                )
                a_blend = (a_det + delta).clamp(-1, 1)
                
                # 计算混合动作的Q值
                q_A_blend = self.policy.critic_A(rep_critic_A, a_blend)
                q_B_blend = self.policy.critic_B(rep_critic_B, a_blend)
                policy_q_blend = torch.min(q_A_blend, q_B_blend).mean()
                
                p_loss = -policy_q_blend
            else:
                policy_q = self.policy.Qpolicy(obs_batch)
                p_loss = -policy_q.mean()
            
            self.optimizer[0].zero_grad()
            p_loss.backward()
            self.optimizer[0].step()
            
            if self.scheduler is not None:
                self.scheduler[0].step()
            
            self.policy.soft_update(self.tau)
            
            # 更新梯度经验存储
            if self.use_grad_bank:
                with torch.no_grad():
                    outputs_actor = self.policy.actor_representation(obs_batch)
                    rep_actor = outputs_actor['state']
                
                a_det = self.policy.actor(rep_actor)
                a_det.requires_grad_(True)
                
                outputs_critic_A = self.policy.critic_A_representation(obs_batch)
                rep_critic_A = outputs_critic_A['state'].detach()
                outputs_critic_B = self.policy.critic_B_representation(obs_batch)
                rep_critic_B = outputs_critic_B['state'].detach()
                
                q_val_A = self.policy.critic_A(rep_critic_A, a_det)
                q_val_B = self.policy.critic_B(rep_critic_B, a_det)
                q_val = (q_val_A + q_val_B).mean() / 2.0
                
                new_grad = torch.autograd.grad(q_val, a_det, retain_graph=False)[0]
                
                # 使用批次平均状态表示和梯度更新经验存储
                mean_state_rep = rep_actor.mean(dim=0, keepdim=True).detach()
                mean_grad = new_grad.mean(dim=0, keepdim=True).detach()
                
                self.grad_bank.update(mean_state_rep, mean_grad)
        else:
            p_loss = torch.tensor(0.0)

        # 日志记录
        actor_lr = self.optimizer[0].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer[1].state_dict()['param_groups'][0]['lr']

        info = {
            "Qloss": q_loss.item(),
            "QvalueA": action_q_A.mean().item(),
            "QvalueB": action_q_B.mean().item(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr
        }
        if self.iterations % self.delay == 0:
            info["Ploss"] = p_loss.item()
            if self.use_grad_bank:
                info["ExpNorm"] = torch.norm(self.grad_bank.experience).item()

        return info