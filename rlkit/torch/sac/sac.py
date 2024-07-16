from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class SACTrainer(TorchTrainer):
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        condition_on_preference=True,
        use_vector_q=False,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        plotter=None,
        render_eval_paths=False,
        use_automatic_entropy_tuning=True,
        target_entropy=None,
        alpha=1.0,
        use_gpu=False,
        wandb_instance=None,
        # weight_pref = torch.tensor([0.5, 0.5]).reshape(2, 1).to("cuda")# np.array([0.5, 0.5]) # MORL weights # update
        # weight_pref = torch.tensor(weight_pref).reshape(2, 1).to("cuda")
    ):
        super().__init__()
        self.wandb_instance = wandb_instance  # for passing values to wandb when wandb is initilized in coadapt class
        self._condition_on_preference = condition_on_preference
        self._use_vector_q = use_vector_q
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env.action_space.shape
                ).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self._alpha = alpha

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self._use_gpu = use_gpu

    def train_from_torch(self, batch, scalarize_before_q_loss=False):
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        terminals = batch["terminals"]
        weight_pref = batch["weight_preferences"]

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs=obs,
            pref=weight_pref if self._condition_on_preference else None,
            reparameterize=True,
            return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self._alpha

        pref_arg = [weight_pref] if self._condition_on_preference else []

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions, *pref_arg),
            self.qf2(obs, new_obs_actions, *pref_arg),
        )
        if self._use_gpu:
            q_new_actions = q_new_actions.to("cuda")

        if self._use_vector_q:
            # scalarize Q-value vector by weighting them with the preferences
            q_new_actions = torch.sum(q_new_actions * weight_pref, dim=1, keepdim=True)

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """

        q1_pred = self.qf1(obs, actions, *pref_arg)
        q2_pred = self.qf2(obs, actions, *pref_arg)

        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            obs=next_obs,
            pref=weight_pref if self._condition_on_preference else None,
            reparameterize=True,
            return_log_prob=True,
        )

        target_q_values = (
            torch.min(
                self.target_qf1(next_obs, new_next_actions, *pref_arg),
                self.target_qf2(next_obs, new_next_actions, *pref_arg),
            )
            - alpha * new_log_pi  # broadcasted in case of use_vector_q=True
        )

        # Match dimensionality of reward and target:
        # If Q-network has vector output, it can handle the vector reward
        # If Q-network has scalar output, scalarize the reward using the preference
        if not self._use_vector_q:
            # BEFORE: rewards (bs, 2) @ weight_pref (2, 1) => (bs, 1) # "bs"="batch size"
            # NOW:    torch.sum(rewards (bs, 2) * weight_pref (bs, 2), dim=1) => (bs, 1)
            rewards = torch.sum(rewards * weight_pref, dim=1, keepdim=True)
        if self._use_gpu:
            rewards = rewards.to("cuda")

        q_target = (
            self.reward_scale * rewards  # past rewards
            + (1.0 - terminals) * self.discount * target_q_values  # expected return
        ).detach()

        if scalarize_before_q_loss:
            # if use_vector_q=False, the Q values are already scalars and this is redundant
            q1_pred = torch.sum(q1_pred * weight_pref, dim=1)
            q2_pred = torch.sum(q2_pred * weight_pref, dim=1)
            q_target = torch.sum(q_target * weight_pref, dim=1)

        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q Targets",
                    ptu.get_numpy(q_target),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.get_numpy(log_pi),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy mu",
                    ptu.get_numpy(policy_mean),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std",
                    ptu.get_numpy(policy_log_std),
                )
            )
            if self.use_automatic_entropy_tuning:
                self.eval_statistics["Alpha"] = alpha.item()
                self.eval_statistics["Alpha Loss"] = alpha_loss.item()

        if self.wandb_instance is not None:
            self.wandb_instance.log(
                {
                    "QF1 loss": self.eval_statistics["QF1 Loss"],
                    "QF2 loss": self.eval_statistics["QF2 Loss"],
                    "Policy loss": self.eval_statistics["Policy Loss"],
                }
            )
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
