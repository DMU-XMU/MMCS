# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import os

import matplotlib.pyplot as plt
from critic_objectives import *
from ig_utils import InfoNCE


class Masker(nn.Module):
    def __init__(self, feature_dim, hidden_dim=8192, k=1024):
        super(Masker, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.k = k

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.bn = nn.BatchNorm1d(feature_dim, affine=False)

    def forward(self, f):
        mask_logits = self.bn(self.layers(f))
        z = torch.zeros_like(mask_logits)
        for _ in range(self.k):
            mask_sample = F.gumbel_softmax(mask_logits, dim=1, tau=0.5, hard=False)
            z = torch.maximum(mask_sample, z)
        return z


class Predictor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Predictor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class NoShiftAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU())
        
        self.linear = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.apply(utils.weight_init)
    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.linear(h)
        return h

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        
        self.apply(utils.weight_init)
    def forward(self, obs, std):
        #h = self.trunk(obs)
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist
    
class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        
        self.apply(utils.weight_init)
    def forward(self, obs, action):
        #h = self.trunk(obs)
        h_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class MMCSAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, mask_steps, embed_dim, 
                 critic_hidden_dim,  offline=False, bc_weight=2.5, augmentation=RandomShiftsAug(pad=4),
                 use_bc=True, k_embed=False, use_critic_grads=True, max_action=1.0, num_club_iter=1):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.offline = offline
        self.bc_weight = bc_weight
        self.use_bc = use_bc
        self.use_critic_grads = use_critic_grads
        self.mask_steps = mask_steps
        self.num_cnce_iter = 1

        # models
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        self.actor = Actor(feature_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        # self.masker = Masker(feature_dim).to(device)
        self.masker = Masker(feature_dim, hidden_dim=4*feature_dim).to(device)
        # self.predictor = Predictor(2*feature_dim, action_shape[0], max_action).to(device)
        self.inv_model = InfoNCE(2*feature_dim+action_shape[0], action_shape[0], 1).to(device)
        
        self.critic = Critic(feature_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(feature_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.linears_club_x1x2_cond = nn.ModuleList([
            mlp(feature_dim, embed_dim, embed_dim, layers=1, activation='relu'), 
            mlp(feature_dim, embed_dim, embed_dim, layers=1, activation='relu'),

            mlp(feature_dim, embed_dim, embed_dim, layers=1, activation='relu'),
            mlp(feature_dim, embed_dim, embed_dim, layers=1, activation='relu'),
            mlp(feature_dim+action_shape[0], embed_dim, embed_dim, layers=1, activation='relu'),
        ]).to(device)

        

        self.club_x1x2_cond1 = CLUBInfoNCECritic(
            embed_dim + embed_dim, 
            action_shape[0], 
            hidden_dim=critic_hidden_dim, 
            layers=1, 
            activation='relu'
        ).to(device)

        self.club_x1x2_cond2 = CLUBInfoNCECritic(
            embed_dim + embed_dim, 
            embed_dim, 
            hidden_dim=critic_hidden_dim, 
            layers=1, 
            activation='relu'
        ).to(device)
 
        self.num_club_iter = num_club_iter

        
        # optimizers
        self.encoder_opt = torch.optim.Adam(list(self.encoder.parameters()), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.masker_opt = torch.optim.Adam(list(self.masker.parameters()), lr=lr)

        self.linears_club_x1x2_cond_opt = torch.optim.Adam(list(self.linears_club_x1x2_cond.parameters()), lr=lr)
        self.club_opt = torch.optim.Adam(list(self.club_x1x2_cond1.parameters())+list(self.club_x1x2_cond2.parameters()), lr=lr)

        # data augmentation
        self.aug = augmentation

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.inv_model.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        if not self.use_critic_grads:
            obs = obs.detach()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward.float() + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        # self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        # self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, behavioural_action=None):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_policy_improvement_loss = -Q.mean()

        actor_loss = actor_policy_improvement_loss

        # offline BC Loss
        if self.offline:
            actor_bc_loss = F.mse_loss(action, behavioural_action)
            # Eq. 5 of arXiv:2106.06860
            lam = self.bc_weight / Q.detach().abs().mean()
            if self.use_bc:
                actor_loss = actor_policy_improvement_loss * lam + actor_bc_loss
            else:
                actor_loss = actor_policy_improvement_loss * lam

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_policy_improvement_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            if self.offline:
                metrics['actor_bc_loss'] = actor_bc_loss.item()

        return metrics
    
    def update_encoder_mmcs(self, obs, obs_prime, step, action):
        metrics = dict()

        features = self.encoder(obs)
        features_prime = self.encoder(obs_prime)
        condition = step >= self.mask_steps

        masks_sup = self.masker(features)
        masks_sup_prime = self.masker(features_prime) if condition else torch.ones_like(features_prime)
        features_sup = features * masks_sup
        features_sup_prime = features_prime * masks_sup_prime
        masks_inf_prime = torch.ones_like(masks_sup_prime) - masks_sup_prime
        features_inf_prime = features_prime * masks_inf_prime

        information_suff_loss = self.inv_model(features, features_sup_prime.detach(), action)
        # obs_cat_sup = torch.cat((features, features_sup_prime.detach()), dim=1)
        # behavioural_action_sup = self.predictor(obs_cat_sup)
        # behavioural_action_sup = F.normalize(behavioural_action_sup, dim=1, p=2)
        # action_norm = F.normalize(action, dim=1, p=2)
        # information_suff_loss = F.mse_loss(action_norm, behavioural_action_sup)

        if condition:
            for _ in range(self.num_cnce_iter): 
                y_emb1 = self.linears_club_x1x2_cond[1](features.detach())
                cnce1_loss = self.club_x1x2_cond1(
                    torch.cat([self.linears_club_x1x2_cond[0](features_inf_prime.detach()), y_emb1], dim=1),
                    action.detach()
                )
                y_emb2 = self.linears_club_x1x2_cond[4](torch.cat([features.detach(), action.detach()], dim=1))
                cnce2_loss = self.club_x1x2_cond2(
                    torch.cat([self.linears_club_x1x2_cond[2](features_sup_prime.detach()), y_emb2], dim=1),
                    self.linears_club_x1x2_cond[3](features_inf_prime.detach())
                )
                cnce_loss = cnce1_loss + cnce2_loss
                self.linears_club_x1x2_cond_opt.zero_grad()
                cnce_loss.backward(retain_graph=True)
                self.linears_club_x1x2_cond_opt.step()

            club1_loss = self.club_x1x2_cond1.learning_loss(torch.cat([self.linears_club_x1x2_cond[0](features_inf_prime), y_emb1.detach()], dim=1), action.detach())
            
            club2_loss = self.club_x1x2_cond2.learning_loss(torch.cat([self.linears_club_x1x2_cond[2](features_sup_prime), y_emb2.detach()], dim=1), 
                                                            self.linears_club_x1x2_cond[3](features_inf_prime))
            club_loss = club1_loss + club2_loss

            total_loss = information_suff_loss + club_loss
            self.club_opt.zero_grad()
            self.masker_opt.zero_grad()
            self.encoder_opt.zero_grad()
            total_loss.backward()
            self.club_opt.step()
            self.masker_opt.step()
            self.encoder_opt.step()
        else:
            self.encoder_opt.zero_grad()
            information_suff_loss.backward()
            self.encoder_opt.step()

        return metrics

    
    def pretrain(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_buffer)
        # obs=obs_prime=:([256, 9, 84, 84]) action:([256, 21])
        obs, action, _, _, _, _, obs_prime = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        obs_prime = self.aug(obs_prime.float())
        metrics.update(self.update_encoder_mmcs(obs, obs_prime, step, action.detach()))

        return metrics
    
    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_buffer)
        obs, action, reward, discount, next_obs, _, obs_prime = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        obs = self.encoder(obs).detach()
        
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        if self.offline:
            metrics.update(self.update_actor(obs.detach(), step, action.detach()))
        else:
            metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def plot_obs(self, obs1, obs2, step):
        def show(img):
            npimg = img.numpy().astype(dtype='uint8')
            fig = plt.imshow(np.transpose(npimg, (1,2,0))[:, :, :], interpolation='nearest')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            return fig

        fig = show(obs1[0, :3, :, :].cpu().data +0.5, ).figure
        fig.savefig(os.path.join(os.getcwd(), 'obs1_{}.png'.format(step)), format='png')

        fig = show(obs2[0, :3, :, :].cpu().data +0.5, ).figure
        fig.savefig(os.path.join(os.getcwd(), 'obs2_{}.png'.format(step)), format='png')

if __name__ == '__main__':
    agent = MMCSAgent(obs_shape=(9, 84, 84), action_shape=(6,), device='cuda', lr=0.3, feature_dim=256,
                 hidden_dim=256, critic_target_tau=0.0, num_expl_steps=5, mask_steps=1000, embed_dim=128, critic_hidden_dim=256,
                 update_every_steps=1000, stddev_schedule='linear(0.1,0.1,25000)', stddev_clip=0.1, use_tb=False,
                 augmentation=RandomShiftsAug(pad=4))
    print("loaded agent successfully!")
