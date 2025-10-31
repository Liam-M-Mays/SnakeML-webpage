import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple


class QNet(nn.Module):
    def __init__(self, data, params, out_dim=3, target=False):
        super().__init__()
        #take a param array and set personal parameter variables here
        self.conv = data["conv"]
        in_dim = int(data["indim"])
        self.prev_state = None
        self.prev_board = None
        self.episodes = 0

        if self.conv:
            mid_dim = 128+in_dim
            self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            )
        else:
            mid_dim = 256
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.v_head = nn.Sequential(
            nn.Linear(mid_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.a_head = nn.Sequential(
            nn.Linear(mid_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )

        if target != True: 
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
            self.t_net = QNet(data, params, target=True)
            self.epsilon = 1
            self.eDecay = float(params["decay"])
            self.replay = ReplayBuffer(int(params["buffer"]))
            self.B = int(params["batch"])
            self.gamma = float(params["gamma"])
    
    def forward(self, x, b):
        # forward defines how inputs flow through layers
        if self.conv:
            body = self.layers(b)
            bodyFlat = self.gap(body).flatten(1)
            head = torch.cat([bodyFlat, x], dim=1)
        else:
            head = self.layers(x)
        v = self.v_head(head)
        a = self.a_head(head)
        aMean = a - torch.mean(a, dim=1, keepdim=True)
        return v + aMean

    ############ SELECT ACTION
    def select_action(self, state_vec, board, target, terminal, device):
        if self.prev_state != None: self.replay.push(self.prev_state, self.prev_board, self.a, target, state_vec, board, terminal)
        if np.random.rand() < self.epsilon:
            self.a = np.random.randint(0, 3)
        else:
            with torch.no_grad():
                x = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
                b = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
                q = self(x, b)
                self.a = int(torch.argmax(q, dim=1).item())

        self.prev_state = state_vec.copy()
        self.prev_board = board.copy()
        if terminal:
            self.episodes += 1
        if len(self.replay) > self.B: 
            self.adjust(device)
        self.epsilon = max(0.1, self.epsilon*self.eDecay)
        return self.a, self.episodes

    ######## TRAIN METHOD
    def adjust(self, device, grad_clip=10.0):

        self.train()
        states, boards, actions, rewards, next_states, next_boards, dones = self.replay.sample(self.B, device=device)
        
        Q_v = self(states, boards)
        q_taken = Q_v.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self(next_states, next_boards).argmax(dim=1, keepdim=True)
            next_q = self.t_net(next_states, next_boards).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (~dones)

        #loss = F.mse_loss(q_taken, target)
        loss = F.smooth_l1_loss(q_taken, target)

        self.optimizer.zero_grad(set_to_none=True)   # clear old gradients
        loss.backward()         # autograd: d(loss)/d(params)
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()        # optimizer applies update
        if self.episodes == 50:
            with torch.no_grad():
                #self.t_net.load_state_dict(self.state_dict(), strict = False)
                sd = self.state_dict()
                online_sd = {k: v for k, v in sd.items() if not k.startswith("t_net.")}
                self.t_net.load_state_dict(online_sd)
        return float(loss.item()), q_taken
    
    def save(self, modelPath):
        self.episodes = 0
        torch.save({"Live":self.state_dict(), "optim":self.optimizer.state_dict()}, modelPath)



##################### Q TABLE
Transition = namedtuple('Transition', 'state board action reward next_state next_board done')
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=int(capacity))

    def push(self, state, board, action, reward, next_state, next_board, done):
        s = np.array(state, dtype=np.float32, copy=True)
        b = np.array(board, dtype=np.float32, copy=True)
        ns = np.array(next_state, dtype=np.float32, copy=True)
        nb = np.array(next_board, dtype=np.float32, copy=True)
        self.buf.append(Transition(s, b, int(action), float(reward), ns, nb, bool(done)))
    
    def __len__(self):
        return len(self.buf)
    
    def sample(self, batch_size, device):
        batch = random.sample(self.buf, batch_size)
        states      = torch.as_tensor(np.array([b.state       for b in batch]), dtype=torch.float32, device=device)
        boards      = torch.as_tensor(np.array([b.board     for b in batch]), dtype=torch.float32, device=device)
        actions     = torch.as_tensor(np.array([b.action      for b in batch]), dtype=torch.long,    device=device)   # Long for gather()
        rewards     = torch.as_tensor(np.array([b.reward      for b in batch]), dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array([b.next_state  for b in batch]), dtype=torch.float32, device=device)
        next_boards = torch.as_tensor(np.array([b.next_board for b in batch]), dtype=torch.float32, device=device)
        dones       = torch.as_tensor(np.array([b.done        for b in batch]), dtype=torch.bool,    device=device)
        return states, boards, actions, rewards, next_states, next_boards, dones
    