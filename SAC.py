import gymnasium as gym
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np

def main():
    warmup_steps = 100_000
    total_steps = 0

    env = gym.make('HumanoidStandup-v5', render_mode = "human")

    agent = Agent(alpha=1e-4, beta=1e-4, input_dims=env.observation_space.shape, env=env, gamma=0.995, n_actions=env.action_space.shape[0],
                  max_size=1_000_000, tau=0.005, batch_size=512)

    
    # Load previous scores if they exista
    start_episode, score_history = agent.load_checkpoint()

    n_games = start_episode + 100_000

    for episode in range(start_episode, n_games):
        observation, info = env.reset()
        state = observation
        done = False
        score = 0

        # initialize losses
        critic_1_loss = None
        critic_2_loss = None
        actor_loss = None
        

        while not done:

            # Select action
            action = agent.choose_action(state, warmup=warmup_steps)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Torso height
            torso_height = env.unwrapped.data.qpos[2]

            # Get torso body
            torso_id = env.unwrapped.model.body("torso").id

            # Torso's local Z axis in world coordinates
            torso_z_axis = env.unwrapped.data.xmat[torso_id].reshape(3, 3)[:, 2]

            # 1.0 = perfectly vertical
            # 0.0 = horizontal
            uprightness = torso_z_axis[2]

            # Height reward
            height_reward = np.clip((torso_height - 1.0) / 0.5, 0.0, 1.0)

            # Upright reward
            upright_reward = np.clip(uprightness, 0.0, 1.0)

            # Combined reward
            reward += 10.0 * height_reward
            reward += 50.0 * upright_reward

            agent.remember(state, action, reward, next_state, terminated)

            losses = agent.learn(warmup=warmup_steps)

            state = next_state
            total_steps += 1
            score += reward

            if losses is not None:
                critic_1_loss, critic_2_loss, actor_loss = losses

        # Save score after each episode
        score_history.append(score)

        print(f'Episode {episode:4d}, Score: {score:.1f}')

        if episode % 20 == 0 and critic_1_loss is not None:
            print(
                f"  Critic1 Loss: {critic_1_loss:.5f} | "
                f"Critic2 Loss: {critic_2_loss:.5f} | "
                f"Actor Loss: {actor_loss:.5f} | "
                f"Replay Size: {agent.memory.mem_cntr}"
            )

        if (episode + 1) % 1000 == 0:
            agent.save_checkpoint( episode + 1, score_history ) 
            print( f"*** Checkpoint saved at episode " f"{episode + 1} ***" )

    agent.save_checkpoint(n_games, score_history)
    print("Training complete. Final checkpoint saved.")

    # ---------- Plot after training finishes ----------
    window = 100
    if len(score_history) >= window:
        moving_avg = np.convolve(
            score_history,
            np.ones(window) / window,
            mode='valid'
        )

        plt.figure(figsize=(10, 5))
        plt.plot(score_history, alpha=0.3, label="Episode Reward")
        plt.plot(
            range(window - 1, len(score_history)),
            moving_avg,
            linewidth=2,
            label=f"{window}-Episode Average"
        )

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("SAC Training on HumanoidStandup-v5")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Not enough episodes for moving average.")

if __name__ == '__main__':
    main()
