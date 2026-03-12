import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FORCED_ENV_ID = "Pendulum-v1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gymnasium simulation without PyBullet")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "ddpg"],
        help="Algorithm to run",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--total-timesteps", type=int, default=5000, help="Total environment steps")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment using human mode",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record episodes to video files",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train selected policy while running episodes",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1024,
        help="Number of collected transitions before each PPO update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for DDPG replay updates",
    )
    parser.add_argument(
        "--ddpg-memory-size",
        type=int,
        default=100000,
        help="Replay buffer size for DDPG",
    )
    parser.add_argument(
        "--ddpg-initial-random-steps",
        type=int,
        default=1000,
        help="Initial random steps for DDPG exploration",
    )
    parser.add_argument(
        "--ddpg-tau",
        type=float,
        default=5e-3,
        help="Soft update factor for DDPG target networks",
    )
    parser.add_argument(
        "--ddpg-gamma",
        type=float,
        default=0.99,
        help="Discount factor for DDPG",
    )
    parser.add_argument(
        "--ddpg-ou-theta",
        type=float,
        default=0.15,
        help="OU noise theta for DDPG",
    )
    parser.add_argument(
        "--ddpg-ou-sigma",
        type=float,
        default=0.2,
        help="OU noise sigma for DDPG",
    )
    parser.add_argument(
        "--ddpg-actor-lr",
        type=float,
        default=3e-4,
        help="Actor learning rate for DDPG",
    )
    parser.add_argument(
        "--ddpg-critic-lr",
        type=float,
        default=1e-3,
        help="Critic learning rate for DDPG",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=os.path.join(CURRENT_DIR, "gym_metrics.txt"),
        help="Path to txt file for episode metrics",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "videos", "gym"),
        help="Directory for saved videos",
    )
    return parser.parse_args()


def make_env(
    render: bool,
    record_video: bool,
    video_dir: str,
    total_episodes: int,
    algo: str,
):
    # RecordVideo needs frames from rgb_array mode.
    render_mode = "rgb_array" if record_video else ("human" if render else None)
    env = gym.make(FORCED_ENV_ID, render_mode=render_mode)

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: episode_id == max(0, total_episodes - 1),
            name_prefix=f"{FORCED_ENV_ID.replace('/', '_')}_{algo}",
        )
    return env


def build_ppo_agent(env):
    from src.ppo import PPOAgent

    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("PPO policy in this file supports only continuous Box action spaces.")

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("PPO policy in this file supports only Box observation spaces.")

    # Always disable checkpoint loading in this runner.
    PPOAgent._try_load_checkpoint = lambda self: None

    n_inputs = int(np.prod(env.observation_space.shape))
    n_actions = int(np.prod(env.action_space.shape))
    return PPOAgent(n_inputs=n_inputs, n_actions=n_actions)


def build_ddpg_agent(env, args):
    from src.ddpg import DDPGCartPoleAgent

    return DDPGCartPoleAgent(
        env=env,
        memory_size=args.ddpg_memory_size,
        batch_size=args.batch_size,
        gamma=args.ddpg_gamma,
        tau=args.ddpg_tau,
        initial_random_steps=args.ddpg_initial_random_steps,
        ou_noise_theta=args.ddpg_ou_theta,
        ou_noise_sigma=args.ddpg_ou_sigma,
        actor_lr=args.ddpg_actor_lr,
        critic_lr=args.ddpg_critic_lr,
    )


def get_action(env, obs, agent=None, algo="ppo", step_idx=0, explore=False):
    if algo == "ppo":
        raw_action, _, _ = agent.select_action(obs)
        if isinstance(env.action_space, gym.spaces.Box):
            return np.clip(raw_action, env.action_space.low, env.action_space.high)
        return raw_action

    env_action, _ = agent.select_action(obs, explore=explore, step_idx=step_idx)
    return env_action


def init_metrics_file(metrics_file: str):
    metrics_dir = os.path.dirname(metrics_file)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("episode,reward,actor_loss_mean,critic_loss_mean\n")


def append_metrics(metrics_file: str, episode: int, reward: float, actor_mean: float, critic_mean: float):
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(f"{episode},{reward:.6f},{actor_mean:.6f},{critic_mean:.6f}\n")


def record_final_episode(agent, video_dir: str, max_steps: int, algo: str):
    eval_env = make_env(render=False, record_video=True, video_dir=video_dir, total_episodes=1, algo=algo)
    try:
        obs, info = eval_env.reset()
        eval_reward = 0.0
        for _ in range(max_steps):
            action = get_action(eval_env, obs, agent=agent, algo=algo, step_idx=0, explore=False)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            eval_reward += float(reward)
            if terminated or truncated:
                break
        print(f"Final eval episode reward={eval_reward:.2f}")
    finally:
        eval_env.close()


def main():
    args = parse_args()
    init_metrics_file(args.metrics_file)

    algo_name = args.algo.lower()
    record_after_training = args.train and args.record_video
    if record_after_training:
        print("Video recording during training is disabled for speed.")
        print("A single final evaluation episode will be recorded after training.")

    env = make_env(
        args.render,
        (args.record_video and not args.train),
        args.video_dir,
        args.episodes,
        algo_name,
    )

    if algo_name == "ppo":
        agent = build_ppo_agent(env)
    else:
        agent = build_ddpg_agent(env, args)

    if agent.device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"{algo_name.upper()} device: {agent.device} ({gpu_name})")
    else:
        print(f"{algo_name.upper()} device: {agent.device}")
    print(f"Environment: {FORCED_ENV_ID} | Algorithm: {algo_name.upper()}")

    effective_rollout_steps = args.rollout_steps
    if algo_name == "ppo":
        print(f"Rollout steps: requested={args.rollout_steps}, effective={effective_rollout_steps}")
    else:
        print(
            f"DDPG replay: batch_size={args.batch_size} | memory_size={args.ddpg_memory_size} | "
            f"initial_random_steps={args.ddpg_initial_random_steps}"
        )

    updates = 0
    global_steps = 0
    episode = 0
    episode_steps = 0
    episode_reward = 0.0
    reward_history = []
    episode_actor_losses = []
    episode_critic_losses = []
    last_obs_for_bootstrap = None

    obs, info = env.reset()
    if algo_name == "ddpg":
        agent.reset_noise()

    try:
        for _ in range(args.total_timesteps):
            state = obs
            if algo_name == "ppo":
                raw_action, log_prob, value = agent.select_action(state)
                action = np.clip(raw_action, env.action_space.low, env.action_space.high)

                # Use the executed action for stored on-policy data.
                if args.train:
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    action_t = torch.as_tensor(action, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    with torch.no_grad():
                        _, log_prob_t, value_t, _ = agent.forward_pass(state_t, action_t)
                    log_prob = float(log_prob_t.squeeze(0).cpu().item())
                    value = float(value_t.squeeze(0).cpu().item())
            else:
                action, action_for_buffer = agent.select_action(
                    state,
                    explore=args.train,
                    step_idx=global_steps,
                )

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_steps += 1
            global_steps += 1

            done = terminated or truncated or (episode_steps >= args.max_steps)

            if args.train:
                if algo_name == "ppo":
                    agent.memory.store(state, action, value, reward, log_prob, done)
                    last_obs_for_bootstrap = None if done else next_obs

                    if len(agent.memory.states) >= effective_rollout_steps:
                        actor_loss, critic_loss = agent.ppo_update(last_obs=last_obs_for_bootstrap)
                        episode_actor_losses.append(float(actor_loss))
                        episode_critic_losses.append(float(critic_loss))
                        updates += 1
                        print(
                            f"PPO update {updates} | steps={global_steps} | "
                            f"actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f}"
                        )
                else:
                    agent.store_transition(state, action_for_buffer, float(reward), next_obs, done)
                    if len(agent.memory) >= args.batch_size:
                        actor_loss, critic_loss = agent.update_model()
                        episode_actor_losses.append(float(actor_loss))
                        episode_critic_losses.append(float(critic_loss))
                        updates += 1
                        if updates % 100 == 0:
                            print(
                                f"DDPG update {updates} | steps={global_steps} | "
                                f"actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f}"
                            )

            obs = next_obs

            if done:
                episode += 1
                reward_history.append(episode_reward)
                mean_reward = float(np.mean(reward_history[-100:]))

                if episode_actor_losses:
                    actor_loss_mean = float(np.mean(episode_actor_losses))
                    critic_loss_mean = float(np.mean(episode_critic_losses))
                else:
                    actor_loss_mean = np.nan
                    critic_loss_mean = np.nan

                append_metrics(args.metrics_file, episode, episode_reward, actor_loss_mean, critic_loss_mean)
                print(
                    f"Episode {episode}/{args.episodes} | steps={episode_steps} | "
                    f"reward={episode_reward:.2f} mean={mean_reward:.2f}"
                )

                if episode >= args.episodes:
                    break

                obs, info = env.reset()
                episode_steps = 0
                episode_reward = 0.0
                episode_actor_losses = []
                episode_critic_losses = []
                if algo_name == "ddpg":
                    agent.reset_noise()

        if args.train and algo_name == "ppo" and len(agent.memory.states) > 0:
            actor_loss, critic_loss = agent.ppo_update(last_obs=last_obs_for_bootstrap)
            updates += 1
            print(
                f"Final PPO update {updates} | steps={global_steps} | "
                f"actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f}"
            )
    finally:
        env.close()

    if record_after_training:
        record_final_episode(agent, args.video_dir, args.max_steps, algo_name)

    if args.record_video:
        print(f"Video saved under: {args.video_dir}")
    print(f"Metrics saved to: {args.metrics_file}")


if __name__ == "__main__":
    main()
