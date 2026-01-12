import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(x, weight=0.999):
    """Exponential Moving Average"""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    smoothed = []
    last = x[0]
    for v in x:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed, dtype=np.float32)

def rolling_average(x, window=300):
    """Rolling average for winrate to make it smooth"""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def main():
    data = np.load("ddqn2_frontier_stats_10x10_9m.npz")
    rewards    = data["rewards"]
    steps      = data["steps"]
    open_cells = data["open_cells"]
    progress   = data["progress"]
    guess      = data["guess"]

    episodes = np.arange(1, len(rewards) + 1)

    # ========= 1) Reward  =========
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Raw reward")
    plt.plot(episodes, smooth_curve(rewards), label="Smoothed reward")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("DDQN Minesweeper: Total Reward")
    plt.grid(True); plt.legend()

    # ========= 2) Opened Cells =========
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, open_cells, alpha=0.3, label="Raw open cells")
    plt.plot(episodes, smooth_curve(open_cells), label="Smoothed open cells")
    plt.xlabel("Episode"); plt.ylabel("Opened Cells")
    plt.title("DDQN Minesweeper: Opened Cells per Episode")
    plt.grid(True); plt.legend()

    # ========= 3) Progress / Guess Ratio =========
    total_moves = progress + guess
    total_moves = np.where(total_moves == 0, 1, total_moves)
    progress_ratio = progress / total_moves
    guess_ratio    = guess / total_moves

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, smooth_curve(progress_ratio), label="Progress ratio")
    plt.plot(episodes, smooth_curve(guess_ratio), label="Guess ratio")
    plt.xlabel("Episode"); plt.ylabel("Ratio")
    plt.title("DDQN Minesweeper: Progress vs Guess Ratio")
    plt.grid(True); plt.legend()

    # ========= 4) Win Rate =========
    WIN_THRESHOLD = 91  # 100 - 9
    wins = (open_cells >= WIN_THRESHOLD).astype(np.float32)

    winrate_smooth = rolling_average(wins, window=950)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes[:len(winrate_smooth)],
             winrate_smooth * 100,       
             label="Win Rate (%, rolling avg 300)",
            color="green")

    plt.xlabel("Episode")
    plt.ylabel("Win Rate (%)")
    plt.title("DDQN Minesweeper: Win Rate (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()