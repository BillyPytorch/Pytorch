import torch as T, numpy as np
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork, ReplayBuffer
import os
import torch.optim as optim

class Agent:
    def __init__(self, alpha=1e-5, beta=1e-5, input_dims=(4, 84, 84), env=None, gamma=0.995, n_actions=3, 
                 max_size=100_000 , tau=0.005, batch_size=256): #TODO max_size change

        # SAC hyterparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Replay buffer
        self.memory = ReplayBuffer(max_size=max_size, input_shape=input_dims, n_actions=n_actions)

        #Actor Network
        self.actor = ActorNetwork(alpha=alpha, input_dims=input_dims, n_actions=n_actions, name='actor')

        # 1. Inside Agent.__init__:
        self.action_dim = env.action_space.shape[0]
        self.target_entropy = -float(self.action_dim) # For 1D action space, target_entropy = -3.0 #TODO
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-5)
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

    @T.no_grad()
    def choose_action(self, observation, warmup=20_000):
        if self.memory.mem_cntr < warmup:
            return self.env.action_space.sample()
        
        
        state = T.tensor(
            observation,
            dtype=T.float32,
            device=self.actor.device
        ).unsqueeze(0) / 255.0

        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        #actions[0, 2] = 0.0
        return actions.detach().cpu().numpy()[0]

    @T.no_grad()
    def choose_actions(self, observations, warmup=20_000):

        # One random action for each environment during warmup
        if self.memory.mem_cntr < warmup:
            return np.array([
                self.env.action_space.sample()
                for _ in range(len(observations))], dtype=np.float32)

        # observations shape:
        # (num_envs, 4, 84, 84)
        state = T.tensor(
            observations,
            dtype=T.float32,
            device=self.actor.device
        ) / 255.0

        # Actor processes all environments as one batch
        actions, _ = self.actor.sample_normal(
            state,
            reparameterize=False
        )

        # actions shape:
        # (num_envs, 3)
        return actions.detach().cpu().numpy()

    def remember(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            self.memory.store_transition(
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                dones[i]
            )

    # Update target critic networks with soft update
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def learn(self, warmup=20_000):
        if self.memory.mem_cntr < warmup:
            return

        # Sample batch from replay buffer 
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Convert numpy arrays to tensors
        state = state.to(self.actor.device)
        new_state = new_state.to(self.actor.device)
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
            alpha = self.log_alpha.exp()

            target = reward + self.gamma * (1 - done) * (q_next - alpha * next_log_probs)

        # Current critic estimates
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        # Mean squared error loss
        critic_1_loss = F.mse_loss(q1, target)
        critic_2_loss = F.mse_loss(q2, target)

        # Optimize critic 1
        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward() # derivative (chain rule) going backwards through the layers (how much each parameter affects loss)
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_1.optimizer.step()

        # Optimizer critic 2
        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
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
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_probs - q_policy).mean()

        # Update Actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update Entropy Alpha
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with T.no_grad():
            self.log_alpha.clamp_(
                T.log(T.tensor(0.05, device=self.actor.device)),
                T.log(T.tensor(1.0, device=self.actor.device)))

        # Current alpha value
        self.entropy_alpha = self.log_alpha.exp().item()

        for param in self.critic_1.parameters():
            param.requires_grad = True

        for param in self.critic_2.parameters():
            param.requires_grad = True

            # Soft-update target critics
        self.update_network_parameters()

        return critic_1_loss.item(), critic_2_loss.item(), actor_loss.item()


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

            'memory': { 
                'state_memory': self.memory.state_memory, 
                'new_state_memory': self.memory.new_state_memory, 
                'action_memory': self.memory.action_memory, 
                'reward_memory': self.memory.reward_memory, 
                'terminal_memory': self.memory.terminal_memory, 
                'mem_cntr': self.memory.mem_cntr
                }
            }

        # Always overwrite the same checkpoint
        checkpoint_file = r"C:\Users\Billy\Documents\sac_checkpoint.pt"

        # Save to a temporary file first
        temp_file = r"C:\Users\Billy\Documents\sac_checkpoint.tmp"

        T.save(checkpoint, temp_file)

        # Atomically replace the old checkpoint
        os.replace(temp_file, checkpoint_file)

        print(f"Checkpoint overwritten: Episode {episode}")

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

        self.log_alpha.data.clamp_(
            T.log(T.tensor(0.05, device=self.actor.device)),
            T.log(T.tensor(1.0, device=self.actor.device))
        ) #TODO added clip

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


