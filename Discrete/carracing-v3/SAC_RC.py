import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation
from agent import Agent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

train_start = time.time()
total_steps = 0

WARMUP_STEPS = 20_000
MAX_EPISODE_STEPS = 1_000
render = False # Human
NUM_ENVS = 1 if render else 20

class CurvatureWrapper(gym.Wrapper):

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        env = self.env.unwrapped
        curvature = self.get_curvature()
        info["curvature_ahead"] = curvature

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        curvature = self.get_curvature()
        info["curvature_ahead"] = curvature

        return obs, reward, terminated, truncated, info

    def get_curvature(self):
        env = self.env.unwrapped

        if not hasattr(env, "track") or not env.track: return 0.0

        track = env.track

        # Car position
        car_x = env.car.hull.position.x
        car_y = env.car.hull.position.y

        car_pos = np.array([car_x, car_y])

        # Track center points
        track_xy = np.array([
            [point[2], point[3]]
            for point in track
        ])

        # Find nearest track point
        distances = np.linalg.norm(track_xy - car_pos, axis=1)
        current_idx = np.argmin(distances)

        # Look ahead on the track
        speed = np.linalg.norm(env.car.hull.linearVelocity)
        lookahead = int(np.clip(speed * 0.1, 1, 15))
        
        i0 = current_idx
        i1 = (current_idx + lookahead) % len(track)

        # Direction vectors
        p0 = track_xy[i0]
        p1 = track_xy[(i0 + 5) % len(track)]

        p2 = track_xy[i1]
        p3 = track_xy[(i1 + 5) % len(track)]

        v1 = p1 - p0
        v2 = p3 - p2

        # Direction angles
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])

        # Difference between directions
        angle_diff = angle2 - angle1

        # Wrap to [-pi, pi]
        angle_diff = np.arctan2(
            np.sin(angle_diff),
            np.cos(angle_diff)
        )

        # Absolute curvature
        curvature = abs(angle_diff)

        # Normalize roughly to 0-1
        curvature = np.clip(curvature / (np.pi / 2), 0.0, 1.0)
        #print(f"Speed: {speed:.1f} | Lookahead: {lookahead} | Curvature: {curvature}")

        return float(curvature)


class CarRacingInfoWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.off_track_steps = 0
        self.episode_steps = 0

        # Register keys on reset so VectorEnv tracks them
        info["speed"] = 0
        info["wheels_on_track"] = 4
        info["off_track_steps"] = 0
        info["curvature_ahead"] = 0
        info["tiles_visited"] = 0
        info["total_tiles"] = 0
        info["episode_steps"] = 0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        
        env = self.env.unwrapped

        vx = self.env.unwrapped.car.hull.linearVelocity[0]
        vy = self.env.unwrapped.car.hull.linearVelocity[1]
        speed = np.sqrt(vx * vx + vy * vy)

        wheels_on_track = sum(
            len(wheel.tiles) > 0
            for wheel in self.env.unwrapped.car.wheels)

        if wheels_on_track == 0:
            self.off_track_steps += 1
        else:
            self.off_track_steps = 0

        if self.off_track_steps >= 80:
            terminated = True

        if env.tile_visited_count >= len(env.track):
            terminated = True

        if self.episode_steps >= MAX_EPISODE_STEPS:
            truncated = True

        info["speed"] = speed
        info["wheels_on_track"] = wheels_on_track
        info["off_track_steps"] = self.off_track_steps
        info["tiles_visited"] = env.tile_visited_count
        info["total_tiles"] = len(env.track)
        info["episode_steps"] = self.episode_steps

        return obs, reward, terminated, truncated, info


def make_env():
    #env = gym.make("CarRacing-v3", continuous=True)
    env = gym.make("CarRacing-v3", continuous=True, render_mode='human' if render else None)
    env = CarRacingInfoWrapper(env)
    env = CurvatureWrapper(env)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    return env


def make_vector_env(num_envs=NUM_ENVS):
    if render:
        return gym.vector.SyncVectorEnv([lambda: make_env() for _ in range(num_envs)])
    else:
        #return gym.vector.SyncVectorEnv([lambda: make_env() for _ in range(num_envs)])
        return gym.vector.AsyncVectorEnv([lambda: make_env() for _ in range(num_envs)])


def main():
    envs = make_vector_env()
    single_env = make_env()

    agent = Agent(
        alpha=3e-5,
        beta=3e-5,
        input_dims=(4, 84, 84),
        env=single_env,
        gamma=0.995,
        n_actions=single_env.action_space.shape[0],
        max_size=100_000,
        tau=0.005,
        batch_size=128
    )

    start_episode, score_history = agent.load_checkpoint()
    target_episodes = start_episode + 100_000

    observations, infos = envs.reset()
    states = np.asarray(observations, dtype=np.uint8)

    episode_scores = np.zeros(NUM_ENVS)
    shaped_scores = np.zeros(NUM_ENVS)
    episode_steps = np.zeros(NUM_ENVS, dtype=int)

    total_speed = np.zeros(NUM_ENVS)
    speed_count = np.zeros(NUM_ENVS, dtype=int)
    total_gas = np.zeros(NUM_ENVS)
    total_brake = np.zeros(NUM_ENVS)
    total_steering = np.zeros(NUM_ENVS)
    total_off_track = np.zeros(NUM_ENVS)
    action_count = np.zeros(NUM_ENVS, dtype=int)

    total_steps = 0
    losses = None

    while start_episode < target_episodes:

        actions = agent.choose_actions(states, warmup=WARMUP_STEPS)
        actions = np.clip(actions,[-1, 0, 0], [1, 1, 1])

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        next_states = np.asarray(next_obs, dtype=np.uint8)
        dones = np.logical_or(terminated, truncated)

        shaped_rewards = rewards.copy()
        for i in range(NUM_ENVS):
            speed = infos["speed"][i]
            wheels = infos["wheels_on_track"][i]
            off_track = infos["off_track_steps"][i]
            tiles_visited = infos["tiles_visited"][i]
            total_tiles = infos["total_tiles"][i]
            episode_steps[i] = infos["episode_steps"][i]
            on_track = wheels > 0

            #episode_steps[i] += 1

            steering = abs(actions[i][0])
            gas = actions[i][1]
            brake = actions[i][2]
            curvature = infos["curvature_ahead"][i]
   
            if on_track:
                total_speed[i] += speed
                speed_count[i] += 1
                total_gas[i] += gas
                total_brake[i] += brake
                total_steering[i] += steering
                action_count[i] += 1
            total_off_track[i] += (wheels == 0)

            shaped_reward = rewards[i]

            # Penalize throttle + brake together
            if brake > 0.2  and gas > 0.2:
                shaped_reward -= (0.2 * gas * brake) 

            # LATER Target speed
            if curvature < 0.2: target_speed = 60.0 # straight
            else: target_speed = 55.0 - 20.0 * curvature  # corners
            target_speed = np.clip(target_speed, 30.0, 60.0)

            if on_track and curvature < 0.2 and speed > 40 and speed <= target_speed:
                shaped_reward += 0.02 * (speed - 40.0)
                #print(0.01 * (speed - 40.0))

            # ==== CURVES ====
            # Slow/Brake on turns
            if 0.4 < curvature < 1.0  and on_track: # and speed > target_speed TODO
                max_safe_speed = 50.0 - (30.0 * curvature)  # Scale safe speed with sharpness
                if speed > max_safe_speed:
                    overspeed = speed - max_safe_speed
                    shaped_reward -= 0.001 * (overspeed ** 2)
                    # Gentle bonus for using brakes when overspeeding in a turn
                    shaped_reward += 0.02 * brake
                    #print(0.02 * brake - 0.001 * (overspeed ** 2))

            # Dont break on no curve
            if curvature < 0.2: shaped_reward -= 0.2 * brake
                
            # ====TRACK====
            # wheels left the track
            if wheels < 4: shaped_reward -= 0.5 * (4 - wheels)
            if wheels == 0: shaped_reward -= 20
            #print(f'TRACK wheels: -{0.1 * (4 - wheels)}')
            #print("OFF TRACK" if wheels == 0 else "", speed)

            if off_track >= 80: shaped_reward -= 50.0 # penalize long time off track

            # ====BEGIN====
            if brake > .2 and speed < 8: shaped_reward -= brake # reduce brake
            #print(f'BEGIN reduce brake: -{brake * 2}')
                
            if speed < 8: shaped_reward -= 0.5 * (8.0 - speed) / 8.0 # penalize low stright speeds
            #print(f'BEGIN low straights: -{2 * (8.0 - speed) / 8.0}')

            shaped_rewards[i] = shaped_reward
            episode_scores[i] += rewards[i]
            shaped_scores[i] += shaped_reward


        agent.remember(
            states,
            actions,
            shaped_rewards,
            next_states,
            dones
        )
        total_steps += NUM_ENVS
        elapsed = time.time() - train_start
        steps_per_sec = total_steps / elapsed

        if total_steps >= WARMUP_STEPS:
            losses = agent.learn(warmup=WARMUP_STEPS)

        for i in range(NUM_ENVS):

            if dones[i]:
                start_episode += 1

                score = episode_scores[i]
                shaped = shaped_scores[i]

                avg_speed = (total_speed[i] / speed_count[i] if speed_count[i] else 0)
                avg_gas = (total_gas[i] / action_count[i] if action_count[i] else 0)
                avg_brake = (total_brake[i] / action_count[i] if action_count[i] else 0)
                avg_steering = (total_steering[i] / action_count[i] if action_count[i] else 0)

                score_history.append(score)
                completed = "COMPLETED" if score > 900 else ""


                print(
                    f"Episode: {start_episode} | "
                    f"Env: {i:02d} | "
                    f"Score: {int(score)} | "
                    f"Shaped: {int(shaped)} | "
                    f"Steps: {episode_steps[i]} | "
                    f"Steps/sec: {steps_per_sec:.1f} | "
                    f"Speed: {avg_speed:.1f} | "
                    f"Gas: {avg_gas:.1f} | "
                    f"Brake: {avg_brake:.1f} | "
                    f"Steering: {avg_steering:.1f} | "
                    f"Off Track: {total_off_track[i]} | "
                    f"{completed}"
                )

                episode_scores[i] = 0
                shaped_scores[i] = 0
                episode_steps[i] = 0
                total_speed[i] = 0
                speed_count[i] = 0
                total_gas[i] = 0
                total_brake[i] = 0
                action_count[i] = 0
                total_off_track[i] = 0

                if start_episode % 100 == 0:
                    agent.save_checkpoint(start_episode, score_history)

                    print(f"*** Checkpoint saved at episode {start_episode} ***")

                    if losses is not None:
                        critic_1_loss, critic_2_loss, actor_loss = losses
                        print(
                            f"Critic1: {critic_1_loss:.1f} | "
                            f"Critic2: {critic_2_loss:.1f} | "
                            f"Actor: {actor_loss:.1f}"
                        )

                    if hasattr(agent, "entropy_alpha"):
                        print(f"Alpha: {agent.entropy_alpha:.6f} | Target Entropy: {agent.target_entropy:.3f}")

        states = next_states

    agent.save_checkpoint(start_episode, score_history)

    envs.close()
    single_env.close()

    if len(score_history) >= 100:
        moving_avg = np.convolve(
            score_history,
            np.ones(100) / 100,
            mode="valid"
        )

        plt.figure(figsize=(10, 5))
        plt.plot(
            score_history,
            alpha=0.3,
            label="Episode Reward"
        )
        plt.plot(
            range(99, len(score_history)),
            moving_avg,
            linewidth=2,
            label="100-Episode Average"
        )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("SAC Training on CarRacing-v3")
        plt.legend()
        plt.grid(True)
        plt.savefig("carracing_training.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()