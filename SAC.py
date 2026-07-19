import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


class ActorNetwork(nn.Module):
    # Input_dims: # observed states
    def __init__(self, alpha, input_dims, max_action,
                 fc1_dims=256, fc2_dims=256,
                 n_actions=2, name='actor',
                 chkpt_dir='tmp/sac'):

        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_sac') # checkpoint file path

        self.max_action = T.tensor(max_action, dtype=T.float32) # make a tensor

        # --- network layers ---
        self.fc1 = nn.Linear(*input_dims, fc1_dims) # y=Wx+b, *input_dims unpacks the dimensions of the input state *removes from tuple
        self.fc2 = nn.Linear(fc1_dims, fc2_dims) # another fully connected layer

        self.mu = nn.Linear(fc2_dims, n_actions) # mean of action distribution (center / best guess)
        self.log_std = nn.Linear(fc2_dims, n_actions) # log standard deviation (controls exploration / uncertainty)

        # --- optimizer ---
        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # new weight = old weight - learning rate * bias-corrected moving average of gradient / (sqrt(bias-corrected moving average of squared gradient) + epsilon)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Run GPU if available
        self.to(self.device) # move to device
        self.max_action = self.max_action.to(self.device) # move num of actions to device

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

    def sample_normal(self, state, reparameterize=True):
        mu, log_std = self.forward(state) # calls results of forward function

        std = log_std.exp() # std-dev of bell curve, std= e^log_std (gradient)

        # guarantees std > 0, which a standard deviation must be. compared to ln(log_std)
        # smoothly converts the network's unrestricted output into a positive width.
        # doesn't have the sharp corner at zero that abs() has.
        # doesn't make +x and -x produce the same standard deviation, as abs() does.

        dist = Normal(mu, std) # bell curve distribution of actions (mean, std-dev)

        # sample
        if reparameterize:
            raw_actions = dist.rsample() # random sample from bell curve
        else:
            raw_actions = dist.sample()

        # squash to valid range
        tanh_actions = T.tanh(raw_actions) # convert to range -1 to 1 (squash)
        action = tanh_actions * self.max_action # scale to max action range

        # log probability -liklihood of the raw_actions
        log_prob = dist.log_prob(raw_actions) # learn formula

        # correction for tanh squashing
        log_prob -= T.log(1 - tanh_actions.pow(2) + 1e-6)

        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))