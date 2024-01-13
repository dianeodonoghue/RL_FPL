from .dqn import *
from .replay_memory import *
import torch.optim as optim
import numpy as np
import math
import random

class trainAgent:

    def __init__(self, player_state):
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 20000
        # Number of random sample taken from memory
        self.BATCH_SIZE = 256
        # Discount factor
        self.GAMMA = 0.999
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get max no. of actions from action space
        player_state = np.array(player_state[:,1:], dtype=float)
        # All player options plus swap
        self.n_players = len(player_state) + 1
        ## all features except the name
        n_features = len(player_state[0]) - 1

        ## Set up DQN for CPU/GPU
        self.policy_net = DQN(self.n_players, n_features).to(self.device)

        # target_net will be updated every n episodes to tell policy_net a better estimate of how far off from convergence
        # Freezes parameters at first so policy and target are the same
        self.target_net = DQN(self.n_players, n_features).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # set target_net in testing mode
        # This is to turn off dropout etc when calculating outpts
        # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        # https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/
        self.target_net.eval()

        # update the parameters based on the computed gradients.
        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.memory = replayMemory()

    def select_action(self, player_info, available_players, player_names, steps_done=None, training=True):
        # batch and color channel
        epsilon = random.random()
        if training:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * steps_done / self.EPS_DECAY)
        else:
            eps_threshold = 0
        
        # follow epsilon-greedy policy
        # Explotation vs exploration
        if epsilon > eps_threshold:
            with torch.no_grad():
                # action recommendations from policy net
                player_stats = torch.tensor(player_info, dtype=torch.float, device=self.device).unsqueeze(dim=0).unsqueeze(dim=0)
                # Returns rewards for each of them
                state_action_values = self.policy_net(player_stats)[0, :]
                # Takes the maximum return
                id = np.argmax(state_action_values.cpu())
        else:
            id = random.choice(available_players)

        ## See if a swap can be made or if its a transfer
        try:
            return player_names[id], id
        except:
            return None, id

    def optimize_model(self):
    # Cant optimise until reaches at least minimum batch size
        if len(self.memory) < self.BATCH_SIZE:
            return
        # Take a random same from memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*[(np.expand_dims(m[0], axis=0), \
                                        [m[1]], m[2], np.expand_dims(m[3], axis=0)) for m in transitions])
        # tensor wrapper
        state_batch = torch.tensor(state_batch, dtype=torch.float, device=self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=self.device)
    
        # If it isn't a final state
        non_final_mask = torch.tensor(tuple(map(lambda s_: s_[0] is not None, next_state_batch)), device=self.device)
        non_final_next_state = torch.cat([torch.tensor(s_, dtype=torch.float, device=self.device).unsqueeze(0) for s_ in next_state_batch if s_[0] is not None])
    
        # prediction from policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    
        # Start with all 0's if no next state, only use reward so don't need target value
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        # Get target values for any non final move
        next_state_values[non_final_mask] = self.target_net(non_final_next_state).max(1)[0].detach()
        # compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # torch.tensor.unsqueeze returns a copy

        # Updates the parameters for the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())