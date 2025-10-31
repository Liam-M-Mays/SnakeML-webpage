import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple


class PPO(nn.Module):
    def __init__(self, data, params, out_dim=3):
        super().__init__()
        #take a param array and set personal parameter variables here
        self.entCoef = 0.05
        self.update_i = 0
        self.conv = data["conv"]
        in_dim = int(data["indim"])
        self.prev_state = None
        self.prev_board = None
        self.buffer = int(params["buffer"])
        self.batch = int(params["batch"])
        self.epoch = int(params["epoch"])
        self.ent_decay_updates = int(params["decay"])
        mid_dim = 256
        if self.conv:
            mid_dim = 256+64
            self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(),
            )

        self.norm = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
        )
        #self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = nn.AdaptiveMaxPool2d(1)
        
        self.v_head = nn.Sequential(
            nn.Linear(mid_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        self.a_head = nn.Sequential(
            nn.Linear(mid_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, out_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.replay = ReplayBuffer(float(params["gamma"]))
        self.episodes = 0

    
    def forward(self, x, b=None):
        # forward defines how inputs flow through layers
        if self.conv:
            body = self.layers(b)
            bodyFlat = self.gap(body).flatten(1)
            norm = self.norm(x)
            head = torch.cat([bodyFlat, norm], dim=1)
        else:
            head = self.norm(x)
        v = self.v_head(head)
        a = self.a_head(head)
        return v, a
    
    ############ SELECT ACTION
    def select_action(self, state_vec, board, target, terminal, device):
        if self.prev_state is not None:
            self.replay.push(self.prev_state, self.prev_board, self.action, target, self.dist.log_prob(self.a).item(), self.crit.squeeze(1).item(), terminal)
        if terminal: 
            self.episodes += 1
        if len(self.replay) >= self.buffer:
            #print(f"[TRAIN] buffer size={len(self.replay)}; starting adjust()")
            self.adjust(device) #TODO epochs and batch size count and do my replay check based on that. pass those values as params for the adjust 
        with torch.no_grad():
            x = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            b = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
            self.crit, act = self(x, b)
            self.dist = torch.distributions.Categorical(logits=act)
            self.a = self.dist.sample()
            self.action = self.a.item()
            self.prev_state = state_vec.copy()
            self.prev_board = board.copy()
            return self.action, self.episodes

    ######## TRAIN METHOD
    def adjust(self, device, grad_clip=0.95):
        batch_size = self.batch
        eps_clip = 0.15
        self.train()
        states, boards, actions, returns, old_probs, old_values = self.replay.sample(device=device)
        adv = (returns - old_values.detach().squeeze(1)) 
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        ret = returns
        ret_n = (ret - ret.mean()) / (ret.std(unbiased=False) + 1e-8)

        idx_all = torch.arange(len(self.replay), device=device)
        #epochin it
        for _ in range(self.epoch):
            perm = idx_all[torch.randperm(len(self.replay))] 
            for i in range(0, len(self.replay), batch_size):
                mb = perm[i:i+batch_size]
                v, a = self(states[mb], boards[mb])

                new_dist = torch.distributions.Categorical(logits=a)
                new_log_prob = new_dist.log_prob(actions[mb])

                #v_loss = F.mse_loss(v.squeeze(1), ret_n[mb])
                v_loss = F.smooth_l1_loss(v.squeeze(1), ret_n[mb])
                ratio = torch.exp(new_log_prob - old_probs[mb].detach())
                a_loss = -torch.min(ratio * adv[mb], (torch.clamp(ratio , 1- eps_clip, 1 + eps_clip) * adv[mb])).mean()
                entropy = new_dist.entropy().mean()
                
                loss = a_loss + 0.5* v_loss - self.entCoef * entropy
                #with torch.no_grad():
                    #print(
                    #f"[PPO] N={mb.numel():3d}  "
                    #f"adv.std={adv[mb].std(unbiased=False).item():.3f}  "
                    #f"ratio μ={ratio.mean().item():.3f} σ={ratio.std(unbiased=False).item():.3f}  "
                    #f"ent={entropy.item():.3f}  vL={v_loss.item():.3f}  aL={a_loss.item():.3f}"
                    #)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)   # clear old gradients
                approx_kl = (old_probs[mb] - new_log_prob).mean()
                if approx_kl.item() > 0.02:
                    break
        ent_coef_start = 0.05
        ent_coef_end = 0.01
        self.update_i += 1
        p = min(1.0, self.update_i / self.ent_decay_updates)
        self.entCoef = (1 - p) * ent_coef_start + p * ent_coef_end
        self.entCoef = max(self.entCoef, ent_coef_end)
        self.replay.clear()
        return 0
    
    def save(self, modelPath):
        self.episodes = 0
        torch.save({"Live":self.state_dict(), "optim":self.optimizer.state_dict()}, modelPath)

Transition = namedtuple('Transition', 'state board action reward prob value')
class ReplayBuffer:
    def __init__(self, gamma):
        self.gamma = gamma
        self.buf = []
        self.epi = []

    def push(self, state, board, action, reward, prob, value, terminal):
        s = np.array(state, dtype=np.float32, copy=True)
        bor = np.array(board, dtype=np.float32, copy=True)
        self.epi.append(Transition(s, bor, action, float(reward), float(prob), float(value)))
        if terminal:
            G = 0.0
            added = 0
            for b in reversed(self.epi):
                G = b.reward + self.gamma * G
                self.buf.append(Transition(b.state, b.board, b.action, float(G), b.prob, b.value))
                added += 1
            self.epi.clear()

    def __len__(self):
        return len(self.buf)
    
    def clear(self):
        self.buf.clear()

    def sample(self, device):
        batch = self.buf[:]
        states      = torch.as_tensor(np.array([b.state       for b in batch]), dtype=torch.float32, device=device)
        boards      = torch.as_tensor(np.array([b.board       for b in batch]), dtype=torch.float32, device=device)
        actions     = torch.as_tensor(np.array([b.action      for b in batch]), dtype=torch.long,    device=device)   
        returns     = torch.as_tensor(np.array([b.reward      for b in batch]), dtype=torch.float32, device=device)
        probs       = torch.as_tensor(np.array([b.prob        for b in batch]), dtype=torch.float32, device=device)
        values      = torch.as_tensor(np.array([b.value       for b in batch]), dtype=torch.float32, device=device).unsqueeze(1)

        return states, boards, actions, returns, probs, values