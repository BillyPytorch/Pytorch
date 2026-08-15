import torch as T
from networks import ActorNetwork, CriticNetwork, ReplayBuffer, RunningMeanStd
import os
import torch.optim as optim

class Agent:
    def __init__(self, alpha=3e-4, beta=3e-4, input_dims=[17], env=None, gamma=0.99, n_actions=None, 
                 max_size=100_000, tau=0.005, batch_size=256):

        # SAC hyterparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Replay buffer
        self.memory = ReplayBuffer(max_size=max_size, input_shape=input_dims, n_actions=n_actions)

        #Actor Network
        self.actor = ActorNetwork(alpha=alpha, input_dims=input_dims, max_action=env.action_space.high, n_actions=n_actions, name='actor')

        self.obs_normalizer = RunningMeanStd(shape=input_dims, epsilon=1e-4, device=self.actor.device)

        # 1. Inside Agent.__init__:
        self.target_entropy = -float(n_actions) * 0.7 # For 1D action space, target_entropy = -1.0
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha)
        self.entropy_alpha = self.log_alpha.exp().item()


        # Critic Networks
        self.critic_1 = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions, name='critic_2')

        # Target Critics
        self.target_critic_1 = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions, name='target_critic_1', create_optimizer=False)
        self.target_critic_2 = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions, name='target_critic_2', create_optimizer=False)

        # copy weights from critic to target critic
        self.update_network_parameters(tau=1)
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)
        self.target_critic_1.eval()
        self.target_critic_2.eval()

    def choose_action(self, observation, warmup=20_000):

        state = T.tensor(
            observation,
            dtype=T.float32,
            device=self.actor.device
        ).unsqueeze(0)

        state = self.obs_normalizer.normalize(state, clip=10.0)

        if self.memory.mem_cntr < warmup:
            return self.env.action_space.sample()

        actions, _ = self.actor.sample_normal(
            state,
            reparameterize=False
        )

        return actions.detach().cpu().numpy()[0]

    def remember(self, state, action, reward, new_state, done):

        raw_state = T.tensor(
            state,
            dtype=T.float32,
            device=self.actor.device
        ).unsqueeze(0)

        raw_new_state = T.tensor(
            new_state,
            dtype=T.float32,
            device=self.actor.device
        ).unsqueeze(0)

        self.obs_normalizer.update(raw_state)
        self.obs_normalizer.update(raw_new_state)

        self.memory.store_transition(
            state,
            action,
            reward,
            new_state,
            done
        )

    # Update target critic networks with soft update
    def update_network_parameters(self, tau=None):
        if tau is None: tau = self.tau

        # Get critic parameters w + b of fc1,fc2, and q
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        # Convert to dictionaries
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)

        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        # Soft update target critic 1
        for name in critic_1_state_dict:
            critic_param = critic_1_state_dict[name]
            target_param = target_critic_1_state_dict[name]
            target_param.data.copy_(tau * critic_param.data + (1 - tau) * target_param.data)    
               
        # Soft update target critic 2    
        for name in critic_2_state_dict:
            critic_param = critic_2_state_dict[name]
            target_param = target_critic_2_state_dict[name]
            target_param.data.copy_(tau * critic_param.data + (1 - tau) * target_param.data)

    def learn(self, warmup=20_000):
        if self.memory.mem_cntr < warmup:
            return

        # Sample batch from replay buffer 
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Convert numpy arrays to tensors
        state = state.to(self.actor.device)
        new_state = new_state.to(self.actor.device)
        state = self.obs_normalizer.normalize(state)
        new_state = self.obs_normalizer.normalize(new_state)
        action = action.to(self.actor.device)
        reward = reward.to(self.actor.device).view(-1,1)
        done = done.float().to(self.actor.device).view(-1,1)

        # Update Critic Networks
        with T.no_grad():

            # Sample next action from actor
            next_actions, next_log_probs = self.actor.sample_normal(new_state, reparameterize=False)

            # Target critic values
            q1_next = self.target_critic_1.forward(new_state, next_actions)
            q2_next = self.target_critic_2.forward(new_state, next_actions)

            # Minimum of the two critics (Clipped Double Q)
            q_next = T.min(q1_next, q2_next)

            # SAC target
            target = reward + self.gamma * (1 - done) * (q_next - self.entropy_alpha * next_log_probs)

        # Current critic estimates
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        # Mean squared error loss
        critic_1_loss = T.nn.functional.mse_loss(q1, target)
        critic_2_loss = T.nn.functional.mse_loss(q2, target)

        # Optimize critic 1
        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1)
        self.critic_1.optimizer.step()

        # Optimizer critic 2
        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1)
        self.critic_2.optimizer.step()

        # Update Actor Network
        for param in self.critic_1.parameters():
            param.requires_grad = False

        for param in self.critic_2.parameters():
            param.requires_grad = False

        new_actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        q1_policy = self.critic_1.forward(state, new_actions)
        q2_policy = self.critic_2.forward(state, new_actions)
        q_policy = T.min(q1_policy, q2_policy)

       # SAC actor loss
        actor_loss = (self.entropy_alpha * log_probs - q_policy).mean()

        # Update Actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update Entropy Alpha
        alpha_loss = -(self.log_alpha * 
                    (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Current alpha value
        self.entropy_alpha = self.log_alpha.exp().item()

        for param in self.critic_1.parameters():
            param.requires_grad = True

        for param in self.critic_2.parameters():
            param.requires_grad = True

            # Soft-update target critics
        self.update_network_parameters()

        return (
            critic_1_loss.item(),
            critic_2_loss.item(),
            actor_loss.item()
        )


    def save_checkpoint(self, episode, score_history):
        os.makedirs('tmp', exist_ok=True)

        checkpoint = {
            'episode': episode,
            'score_history': score_history,

            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor.optimizer.state_dict(),

            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),

            'critic_1_optimizer': self.critic_1.optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2.optimizer.state_dict(),

            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),

            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),

            'obs_normalizer': self.obs_normalizer.state_dict(),

            'memory': { 
                'state_memory': self.memory.state_memory, 
                'new_state_memory': self.memory.new_state_memory, 
                'action_memory': self.memory.action_memory, 
                'reward_memory': self.memory.reward_memory, 
                'terminal_memory': self.memory.terminal_memory, 
                'mem_cntr': self.memory.mem_cntr
                }
            }

        # Temporary file
        temp_file = "C:\\Users\\Billy\\Documents\\sac_checkpoint.tmp"
        T.save(checkpoint, temp_file)

        # Latest checkpoint
        latest_file = "C:\\Users\\Billy\\Documents\\sac_checkpoint.pt"
        os.replace(temp_file, latest_file)

        # Episode-specific checkpoint
        episode_file = f"C:\\Users\\Billy\\Documents\\sac_checkpoint_{episode}.pt"
        T.save(checkpoint, episode_file)

        print(f"Checkpoint saved: Episode {episode}")

    def load_checkpoint(self):
            
        if not os.path.exists(r"C:\Users\Billy\Documents\sac_checkpoint.pt"):
            print("No checkpoint found. Starting fresh.")
            return 0, []

        checkpoint = T.load(r"C:\Users\Billy\Documents\sac_checkpoint.pt",
                            map_location=self.actor.device,
                            weights_only=False)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])

        self.obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])


        # Restore replay buffer
        memory = checkpoint['memory']

        self.memory.state_memory = memory['state_memory']
        self.memory.new_state_memory = memory['new_state_memory']
        self.memory.action_memory = memory['action_memory']
        self.memory.reward_memory = memory['reward_memory']
        self.memory.terminal_memory = memory['terminal_memory']
        self.memory.mem_cntr = memory['mem_cntr']

        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])

        self.target_critic_1.eval()
        self.target_critic_2.eval()

        self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1.optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2.optimizer.load_state_dict(checkpoint['critic_2_optimizer'])

        self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.actor.device))

        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        self.entropy_alpha = self.log_alpha.exp().item()
        print(f"Alpha: {self.entropy_alpha:.6f}")

        print(f"Loaded episode {checkpoint['episode']}")

        return (
            checkpoint['episode'] + 1,
            checkpoint.get('score_history', [])
        )

    def save_models(self):
        print('saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        print('loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()


'''

    def choose_action(self, observation, warmup=20_000):

        state = T.tensor(
            observation,
            dtype=T.float32,
            device=self.actor.device
        ).unsqueeze(0)

        state = self.obs_normalizer.normalize(state, clip=10.0)

        if self.memory.mem_cntr < warmup:
            return self.env.action_space.sample()

        actions, _ = self.actor.sample_normal(
            state,
            reparameterize=False
        )

        return actions.detach().cpu().numpy()[0]


'''


