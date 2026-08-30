import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

# What action the agent should take
class ActorNetwork(nn.Module):
    # Input_dims: # observed states
    # alpha: learning rate
    def __init__(self, alpha, input_dims,
                 fc1_dims=512, fc2_dims=512,
                 n_actions=3, name='actor',
                 chkpt_dir='tmp/sac'):

        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_sac') # checkpoint file path

        # --- network layers ---
        # Input = 96 * 96 * 3 (pixels * rgb) #TODO
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, 2, 1)
        self.gn1 = nn.GroupNorm(4, 32)

        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.gn2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.gn3 = nn.GroupNorm(8, 128)

        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.gn4 = nn.GroupNorm(8, 128)

        conv_output_size = self._get_conv_output(input_dims)

        self.fc1 = nn.Linear(conv_output_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Linear(fc2_dims, n_actions)

        # --- optimizer ---
        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # new weight = old weight - learning rate * bias-corrected moving average of gradient / (sqrt(bias-corrected moving average of squared gradient) + epsilon)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Run GPU if available
        self.to(self.device) # move to device

        # numerical stability
        self.LOG_STD_MIN = -20 # standard min about zero
        self.LOG_STD_MAX = 2 # standard max about 7.4

    def forward(self, state): #TODO
        x = self.conv_layers(state)
        x = T.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)

        log_std = T.clamp(
            log_std,
            self.LOG_STD_MIN,
            self.LOG_STD_MAX
        )

        return mu, log_std

    def conv_layers(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        return F.relu(self.gn4(self.conv4(x)))


    def _get_conv_output(self, input_dims): #TODO
        dummy = T.zeros(1, *input_dims)
        output = self.conv_layers(dummy)
        return int(T.prod(T.tensor(output.shape[1:])))



    # from the state take sample from distribution of actions
    def sample_normal(self, state, reparameterize=True):
        mu, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(mu, std)
        raw = dist.rsample() if reparameterize else dist.sample()

        # tanh squashing
        tanh_raw = T.tanh(raw)

        # --- Correct action scaling ---
        # steering stays [-1,1]
        steering = tanh_raw[:, 0:1]

        # gas & brake scaled to [0,1]
        gas_brake = (tanh_raw[:, 1:] + 1.0) * 0.5

        action = T.cat([steering, gas_brake], dim=1)

        # --- Correct log-prob correction ---
        # SAC tanh correction
        log_prob = dist.log_prob(raw)
        log_prob -= T.log(1 - tanh_raw.pow(2) + 1e-6)

        # affine transform correction for gas/brake scaling
        # a = (tanh(u)+1)/2  → derivative = 0.5
        log_prob[:, 1:] -= T.log(T.tensor(0.5, device=log_prob.device))

        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


# Estimated future reward for a given state + action
class CriticNetwork(nn.Module):
    # beta: learning rate
    def __init__(self, beta, input_dims, n_actions,
                 fc1_dims=512, fc2_dims=512,
                 name='critic',
                 chkpt_dir='tmp/sac',
                 create_optimizer=True):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_sac')

        # Network layers
        # Input = 96 * 96 * 3 (pixels * rgb) TODO
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, 2, 1)
        self.gn1 = nn.GroupNorm(4, 32)

        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.gn2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.gn3 = nn.GroupNorm(8, 128)

        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.gn4 = nn.GroupNorm(8, 128)

        conv_output_size = self._get_conv_output(input_dims) #TODO
        # State input
        self.fc1 = nn.Linear(conv_output_size + n_actions, fc1_dims) #TODO

        # State hidden layer
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)

        # State + action -> Q value (final layer)
        self.q = nn.Linear(fc2_dims, 1)

        # Optimizer
        if create_optimizer:
            self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state, action): #TODO
        x = self.conv_layers(state)
        x = T.flatten(x, start_dim=1)
        state_action = T.cat([x, action], dim=1)
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        return self.q(x)

    def _get_conv_output(self, input_dims): #TODO
        dummy = T.zeros(1, *input_dims)
        output = self.conv_layers(dummy)
        return int(T.prod(T.tensor(output.shape[1:])))


    def conv_layers(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        return F.relu(self.gn4(self.conv4(x)))



    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

# Memory
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        # Memory channels-first: (max_size, 4, 84, 84)
        self.state_memory = T.zeros((self.mem_size, *input_shape), dtype=T.uint8, device='cpu')
        self.new_state_memory = T.zeros((self.mem_size, *input_shape), dtype=T.uint8, device='cpu')
        self.action_memory = T.zeros((self.mem_size, n_actions), dtype=T.float32, device='cpu')
        self.reward_memory = T.zeros(self.mem_size, dtype=T.float32, device='cpu')
        self.terminal_memory = T.zeros(self.mem_size, dtype=T.bool, device='cpu')

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = T.as_tensor(state, dtype=T.uint8)
        self.new_state_memory[index] = T.as_tensor(state_, dtype=T.uint8)
        self.action_memory[index] = T.as_tensor(action, dtype=T.float32)
        self.reward_memory[index] = reward
        self.terminal_memory[index] = bool(done)

        self.mem_cntr += 1

    def sample_buffer(self, batch_size): #TODO
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = T.randint(0, max_mem, (batch_size,))

        states = self.state_memory[batch].float() / 255.0
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch].float() / 255.0
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal