import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot gym training metrics")
    parser.add_argument("--input_name", type=str, default="gym_metrics.txt")
    parser.add_argument("--output_name", type=str, default="graph_result.png")
    parser.add_argument("--smooth-window", type=int, default=1, help="Moving-average window size")
    return parser.parse_args()


def load_metrics(file_path: str):
    episodes = []
    rewards = []
    actor_losses = []
    critic_losses = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            actor_losses.append(float(row["actor_loss_mean"]))
            critic_losses.append(float(row["critic_loss_mean"]))

    episodes = np.asarray(episodes, dtype=np.int32)
    rewards = np.asarray(rewards, dtype=np.float64)
    actor_losses = np.asarray(actor_losses, dtype=np.float64)
    critic_losses = np.asarray(critic_losses, dtype=np.float64)
    return episodes, rewards, actor_losses, critic_losses


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = np.full_like(values, np.nan, dtype=np.float64)
    n = values.shape[0]
    for i in range(n):
        lo = max(0, i - window + 1)
        segment = values[lo : i + 1]
        if np.isfinite(segment).any():
            out[i] = np.nanmean(segment)
    return out


def plot_reward(ax, episodes, rewards, smooth_window: int):
    smoothed = moving_average(rewards, smooth_window)
    ax.plot(episodes, smoothed, label="reward", color="black")

    ax.set_title("Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend(loc="upper left")


def plot_losses(ax, episodes, actor_losses, critic_losses, smooth_window: int):
    actor_smoothed = moving_average(actor_losses, smooth_window)
    critic_smoothed = moving_average(critic_losses, smooth_window)
    ax.plot(episodes, actor_smoothed, label="actor loss", color="blue")
    ax.plot(episodes, critic_smoothed, label="critic loss", color="red")
    ax.set_title("Actor&Critic Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend(loc="upper left")


def save_figure(fig, output_path: str):
    output = Path(output_path)
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Saved graph: {output}")


def main():
    args = parse_args()
    episodes, rewards, actor_losses, critic_losses = load_metrics(args.input_name)
    smooth_window = max(1, int(args.smooth_window))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    plot_reward(axes[0], episodes, rewards, smooth_window)
    plot_losses(axes[1], episodes, actor_losses, critic_losses, smooth_window)
    save_figure(fig, args.output_name)


if __name__ == "__main__":
    main()