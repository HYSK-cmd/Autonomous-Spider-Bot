import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from ppo import PPOAgent
import os

# ===== 모드 선택 =====
MODE = "train"  # "train" 또는 "test" ← 다시 훈련하세요!
MODEL_PATH = "trained_models"
os.makedirs(MODEL_PATH, exist_ok=True)

env = gym.make('LunarLanderContinuous-v3') 

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, buffer_size=1024)

# ===== 훈련 모드 =====
if MODE == "train":
    episode_rewards = []
    actor_losses = []
    critic_losses = []

    print("Training starts")

    state, info = env.reset()
    num_episodes = 50  # 100 → 200 (2배 훈련)

    for episode in range(num_episodes):
        state, rollout_episode_rewards = agent.collect_rollout(env, state)
        avg_episode_reward = np.mean(rollout_episode_rewards) if len(rollout_episode_rewards) > 0 else 0.0
        actor_loss, critic_loss = agent.ppo_update()
        
        episode_rewards.append(avg_episode_reward)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        
        print(f"[{episode+1}/{num_episodes}] reward: {avg_episode_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

    env.close()

    print("\nDone Training!")
    
    # 모델 저장
    torch.save(agent.actor.state_dict(), f'{MODEL_PATH}/actor.pt')
    torch.save(agent.critic.state_dict(), f'{MODEL_PATH}/critic.pt')
    print(f"✓ 모델 저장 완료: {MODEL_PATH}/")

    # 그래프 저장
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(episode_rewards, label='Episode Reward', color='blue', marker='o')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Reward')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(actor_losses, label='Actor Loss', color='red', marker='o')
    axes[1].plot(critic_losses, label='Critic Loss', color='orange', marker='s')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    plt.close()

# ===== 테스트 모드 (비디오 생성) =====
else:
    print("Loading trained model...")
    agent.actor.load_state_dict(torch.load(f'{MODEL_PATH}/actor.pt'))
    agent.critic.load_state_dict(torch.load(f'{MODEL_PATH}/critic.pt'))
    print("✓ 모델 로드 완료")

# ===== 영상 녹화 =====
video_folder = "videos/ppo"
os.makedirs(video_folder, exist_ok=True)

print("\n영상 녹화 시도 중...")
try:
    from gymnasium.wrappers import RecordVideo
    env_video = gym.make('LunarLanderContinuous-v3', render_mode='rgb_array')
    env_video = RecordVideo(env_video, video_folder=video_folder, 
                           episode_trigger=lambda x: x < 5,  # 5개 에피소드 녹화
                           name_prefix='ppo_', disable_logger=True)

    total_reward = 0
    
    # 5개 에피소드 실행
    for ep in range(5):
        state, info = env_video.reset()
        episode_reward = 0
        
        for step in range(1000):
            with torch.no_grad():
                action, _, _ = agent.select_action(state)
                action = action.numpy()
            
            state, reward, terminated, truncated, info = env_video.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                print(f"  - Episode {ep+1} 보상: {episode_reward:.2f}")
                total_reward += episode_reward
                break

    env_video.close()
    print(f"✓ 영상 저장 완료: {video_folder}/")
    print(f"평균 보상: {total_reward/5:.2f}")
    
except Exception as e:
    print(f"✗ 영상 저장 실패: {type(e).__name__}: {str(e)}")
    print("  에이전트 성능 테스트 진행 중...")
    
    try:
        env_test = gym.make('LunarLanderContinuous-v3')
        state, info = env_test.reset()
        episode_reward = 0

        for step in range(1000):
            with torch.no_grad():
                action, _, _ = agent.select_action(state)
                action = action.numpy()
            
            state, reward, terminated, truncated, info = env_test.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        env_test.close()
        print(f"✓ 테스트 완료 | 에피소드 보상: {episode_reward:.2f}")
    except Exception as e2:
        print(f"✗ 테스트 실패: {type(e2).__name__}: {str(e2)}")