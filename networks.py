import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class ActorNetwork(nn.Module):
    # Input_dims: # observed states
    # alpha: learning rate
    def __init__(self, alpha, input_dims, max_action,
                 fc1_dims=512, fc2_dims=512,
                 n_actions=2, name='actor',
                 chkpt_dir='tmp/sac'):

        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_sac') # checkpoint file path

        # --- network layers ---
        self.fc1 = nn.Linear(*input_dims, fc1_dims) # y=Wx+b, *input_dims unpacks the dimensions of the input state *removes from tuple
        self.fc2 = nn.Linear(fc1_dims, fc2_dims) # another fully connected layer

        self.mu = nn.Linear(fc2_dims, n_actions) # mean of action distribution (center / best guess)
        self.log_std = nn.Linear(fc2_dims, n_actions) # log standard deviation (controls exploration / uncertainty)

        # --- optimizer ---
        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # new weight = old weight - learning rate * bias-corrected moving average of gradient / (sqrt(bias-corrected moving average of squared gradient) + epsilon)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Run GPU if available
        self.register_buffer("max_action", T.tensor(max_action, dtype=T.float32)) # move num of actions to device
        self.to(self.device) # move to device

        # numerical stability
        self.LOG_STD_MIN = -20 # standard min about zero
        self.LOG_STD_MAX = 2 # standard max about 7.4

    def forward(self, state):
        x = F.relu(self.fc1(state)) # linear function made non-linear
        x = F.relu(self.fc2(x)) # linear function made non-linear

        mu = self.mu(x) # final linear layer (average/preferred action) mu is the expected result its the mean for distribution of actions (center of bell curve)
        log_std = self.log_std(x) # how much randomness to add (not complete until .exp()) = ln(std) width of bell curve based on current state

        # stabilize std
        log_std = T.clamp(log_std,
                              self.LOG_STD_MIN,
                              self.LOG_STD_MAX) # sets a minimum and maximum limit

        return mu, log_std

    # from the state take sample from distribution of actions
    def sample_normal(self, state, reparameterize=True): 
        mu, log_std = self.forward(state) # calls results of forward function

        std = log_std.exp() # std-dev of bell curve, e**log_std (gradient)

        # guarantees std > 0, which a standard deviation must be. compared to ln(log_std)
        # smoothly converts the network's unrestricted output into a positive width.
        # doesn't have the sharp corner at zero that abs() has.
        # doesn't make +x and -x produce the same standard deviation, as abs() does.

        dist = Normal(mu, std) # bell curve(probability) distribution of actions (mean, std-dev) x is not calculated here (raw_actions)
        # 1 / (std * sqrt(2 * pi)) * e^(-1/2 * ((x - mu) / std) ** 2)

        # sample
        if reparameterize:
            raw_actions = dist.rsample() # random sample of bell curve centered at mu with spread of std = mu + std * rand(0,1)
        else:
            raw_actions = dist.sample()

        # squash to valid range
        tanh_actions = T.tanh(raw_actions) # bound range to -1 through 1 (squash) hyperbolic tangent function prevents the infinity from tan functions e^2 - e^-2 / e^2 + e^-2
        action = tanh_actions * self.max_action # scale to max action range

        # log probability -liklihood of the raw_actions
        log_prob = dist.log_prob(raw_actions) # ln(Normal(mu, std) as raw_actions, makes number easier to work with, changes multiplication into addition

        # correction for tanh squashing
        log_prob -= T.log((1 - tanh_actions.pow(2)).clamp(min=1e-6)) # removes infinity
        log_prob -= T.log(self.max_action.clamp(min=1e-6))

        log_prob = log_prob.sum(dim=1, keepdim=True) # sum across actions, keepdim keeps the same number of dimensions (1,1) instead of (1,) for batch size 1

        return action, log_prob

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))



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
        # State input
        self.fc1 = nn.Linear(input_dims[0]+ n_actions,fc1_dims)

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


    def forward(self, state, action):
    # 1. Concatenate state (256x3) and action (256x1) -> state_action (256x4)
        state_action = T.cat([state, action], dim=1)

        # 2. Pass the combined (256x4) tensor into fc1 (which expects 4 features)
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))

        # 3. Output scalar Q-value
        q_value = self.q(x)

        return q_value


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        #Memory
        self.state_memory = T.zeros((self.mem_size, *input_shape), dtype=T.float32)
        self.new_state_memory = T.zeros((self.mem_size, *input_shape), dtype=T.float32)
        self.action_memory = T.zeros((self.mem_size, n_actions), dtype=T.float32)
        self.reward_memory = T.zeros(self.mem_size, dtype=T.float32)
        self.terminal_memory = T.zeros(self.mem_size, dtype=T.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = T.as_tensor(state, dtype=T.float32)
        self.new_state_memory[index] = T.as_tensor(state_, dtype=T.float32)
        self.action_memory[index] = T.as_tensor(action, dtype=T.float32)
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = T.randint(0, max_mem, (batch_size,), device=self.state_memory.device)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class RunningMeanStd:

    def __init__(self, shape, epsilon=1e-4, device='cpu'):
        self.mean = T.zeros(shape, dtype=T.float64, device=device)
        self.var = T.ones(shape, dtype=T.float64, device=device)
        self.count = T.tensor(float(epsilon), dtype=T.float64, device=device)
        self.device = device

    @T.no_grad()
    def update(self, x):
        x = x.to(self.device).double()

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count

        m2 = (
            m_a +
            m_b +
            delta.pow(2) *
            self.count *
            batch_count /
            total_count
        )

        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @T.no_grad()
    def normalize(self, x, clip=10.0):
        x = x.to(self.device)

        normalized = (
            x.double() - self.mean
        ) / T.sqrt(self.var + 1e-8)

        normalized = T.clamp(
            normalized,
            -clip,
            clip
        )

        return normalized.float()

    def state_dict(self):
        return {
            'mean': self.mean.clone(),
            'var': self.var.clone(),
            'count': self.count.clone()
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean'].to(self.device)
        self.var = state_dict['var'].to(self.device)
        self.count = state_dict['count'].to(self.device)