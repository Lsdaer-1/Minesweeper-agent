import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from minesweeper import MinesweeperDiscreetEnv

# ----------------- Environment and Constant Configuration -----------------

BOARD_SIZE = 10
NUM_MINES  = 9

CLOSED_VAL = -2     # unopened tiles
MINE_VAL   = -1     # mines

NUM_EPISODES = 15000
MAX_STEPS_PER_EPISODE = 120

GAMMA = 0.1

BATCH_SIZE = 64
LR = 0.001

EPS_START = 0.8
EPS_END   = 0.05
EPS_DECAY_STEPS = NUM_EPISODES*2.5

TARGET_UPDATE_STEPS = 1500
REPLAY_CAPACITY = 500000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


# ----------------- Experience Replay -----------------

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


# ----------------- 3 channels CNN -----------------

class MinesweeperDQN(nn.Module):
    def __init__(self, board_size, num_actions):
        super().__init__()
        self.board_size = board_size
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(11, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        conv_out = 128 * board_size * board_size
        self.fc1 = nn.Linear(conv_out, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ----------------- state encoding -----------------

def preprocess_state(raw_board: np.ndarray) -> np.ndarray:
    board = raw_board.astype(np.int32)

    # 1. closed Mask (-2)
    closed_mask = (board == CLOSED_VAL).astype(np.float32)

    # 2. One-Hot (0, 1, ..., 8)
    number_layers = []
    for i in range(9):
        layer = (board == i).astype(np.float32)
        number_layers.append(layer)

    frontier_mask = compute_frontier_mask(raw_board).astype(np.float32)

    all_layers = [closed_mask] + number_layers + [frontier_mask]
    stacked = np.stack(all_layers, axis=0)
    return stacked


def preprocess_state1(raw_board: np.ndarray) -> np.ndarray:
    board = raw_board.astype(np.float32)

    closed_mask = (board == CLOSED_VAL).astype(np.float32)

    numbers = board.copy()
    numbers[numbers < 0] = 0.0
    numbers[numbers > 8] = 8.0
    numbers_norm = numbers / 8.0

    zero_mask = (board == 0).astype(np.float32)

    stacked = np.stack([closed_mask, numbers_norm, zero_mask], axis=0)
    return stacked


# ----------------- frontier mask -----------------

def compute_frontier_mask(raw_state: np.ndarray) -> np.ndarray:
    H, W = raw_state.shape
    closed = (raw_state == CLOSED_VAL)
    open_cells = (raw_state != CLOSED_VAL)

    frontier = np.zeros_like(closed, dtype=bool)

    for i in range(H):
        for j in range(W):
            if not closed[i, j]:
                continue
            # Check if there are open cells in the surrounding 8 neighborhoods.
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if open_cells[ni, nj]:
                            frontier[i, j] = True
                            break
                if frontier[i, j]:
                    break

    return frontier


def get_action_mask_frontier(env,
                             raw_state: np.ndarray,
                             num_actions: int,
                             first_move: bool) -> np.ndarray:
    flat_closed = (raw_state.flatten() == CLOSED_VAL)

    if first_move:
        not_mine = (env.board.flatten() != MINE_VAL)
        mask = flat_closed & not_mine
        return mask

    frontier = compute_frontier_mask(raw_state).flatten()

    if frontier.any():
        return frontier
    else:
        return flat_closed


# ----------------- progress / guess  -----------------

def classify_move(old_state: np.ndarray, new_state: np.ndarray) -> str:
    H, W = old_state.shape

    closed_before = (old_state == CLOSED_VAL)
    closed_after  = (new_state == CLOSED_VAL)
    new_open_mask = closed_before & ~closed_after
    new_count = int(new_open_mask.sum())

    if new_count == 0:
        return "no_change"

    open_before = (old_state != CLOSED_VAL)

    for i in range(H):
        for j in range(W):
            if not new_open_mask[i, j]:
                continue
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if open_before[ni, nj]:
                            return "progress"

    return "guess"


# ----------------- epsilon-greedy + Double DQN  -----------------

def epsilon_by_step(global_step: int) -> float:
    if global_step >= EPS_DECAY_STEPS:
        return EPS_END
    ratio = (EPS_DECAY_STEPS - global_step) / EPS_DECAY_STEPS
    return EPS_END + (EPS_START - EPS_END) * max(0.0, ratio)


def select_action(policy_net,
                  state_np: np.ndarray,
                  global_step: int,
                  num_actions: int,
                  valid_mask: np.ndarray) -> (int, float):
    eps = epsilon_by_step(global_step)

    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return random.randrange(num_actions), eps

    if random.random() < eps:
        action = int(np.random.choice(valid_indices))
        return action, eps
    else:
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor).squeeze(0).cpu().numpy()
        q_masked = q_values.copy()
        q_masked[~valid_mask] = -1e9
        action = int(q_masked.argmax())
        return action, eps


# ----------------- Double DQN  -----------------

def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*transitions)

    state_batch = torch.from_numpy(np.stack(batch.state)).float().to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_state_batch = torch.from_numpy(np.stack(batch.next_state)).float().to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device)

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        online_next_q = policy_net(next_state_batch)
        next_actions = online_next_q.argmax(dim=1, keepdim=True)

        target_next_q = target_net(next_state_batch)
        next_q_values = target_next_q.gather(1, next_actions).squeeze(1)

        target_values = reward_batch + (1.0 - done_batch) * GAMMA * next_q_values

    loss = F.smooth_l1_loss(q_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()



def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = MinesweeperDiscreetEnv(board_size=BOARD_SIZE, num_mines=NUM_MINES)

    raw_obs = env.reset()
    H, W = raw_obs.shape
    assert (H, W) == (BOARD_SIZE, BOARD_SIZE)
    num_actions = env.action_space.n

    print(f"Environment initialization successful: board=({H},{W}), num_actions={num_actions}")
    print(f"device: {device}")

    policy_net = MinesweeperDQN(BOARD_SIZE, num_actions).to(device)
    target_net = MinesweeperDQN(BOARD_SIZE, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(REPLAY_CAPACITY)

    global_step = 0

    all_rewards = []
    all_steps = []
    all_open_cells = []
    all_progress_counts = []
    all_guess_counts = []

    for episode in range(1, NUM_EPISODES + 1):
        raw_state = env.reset()
        state = preprocess_state(raw_state)

        episode_reward = 0.0
        steps_in_episode = 0
        done = False
        first_move = True
        last_eps = EPS_START

        ep_progress = 0
        ep_guess = 0

        while not done and steps_in_episode < MAX_STEPS_PER_EPISODE:
            steps_in_episode += 1
            global_step += 1

            valid_mask = get_action_mask_frontier(
                env, raw_state, num_actions, first_move
            )

            action, eps = select_action(
                policy_net, state, global_step, num_actions, valid_mask
            )
            last_eps = eps

            old_raw_state = raw_state.copy()
            next_raw_state, env_reward, done, info = env.step(action)
            next_state = preprocess_state(next_raw_state)

            # ----- coustom reward -----
            if done:
                if env_reward == 1000:
                    reward = 1   # win
                    move_type = "terminal_win"
                elif env_reward == -100:
                    reward = -1   # lose
                    move_type = "terminal_lose"
                else:
                    reward = 0.0
                    move_type = "terminal"
            else:
                move_type = classify_move(old_raw_state, next_raw_state)
                if move_type == "progress":
                    reward = 0.3
                    ep_progress += 1
                elif move_type == "guess":
                    reward = -0.3
                    ep_guess += 1
                else:
                    reward = -99

            memory.push(state, action, reward, next_state, float(done))
            optimize_model(policy_net, target_net, memory, optimizer)

            raw_state = next_raw_state
            state = next_state
            episode_reward += reward
            first_move = False

            if global_step % TARGET_UPDATE_STEPS == 0:
                target_net.load_state_dict(policy_net.state_dict())

        all_rewards.append(episode_reward)
        all_steps.append(steps_in_episode)
        open_cells_count = int((raw_state != CLOSED_VAL).sum())
        all_open_cells.append(open_cells_count)
        all_progress_counts.append(ep_progress)
        all_guess_counts.append(ep_guess)

        print(
            f"[Train] Ep {episode}/{NUM_EPISODES} | "
            f"steps={steps_in_episode} | reward={episode_reward:.2f} | "
            f"open_cells={open_cells_count} | "
            f"progress={ep_progress} | guess={ep_guess} | eps={last_eps:.3f}"
        )

    model_name = f"ddqn2_frontier_3ch_{BOARD_SIZE}x{BOARD_SIZE}_{NUM_MINES}m.pt"
    torch.save(policy_net.state_dict(), model_name)
    print(f"Training complete. Model saved as {model_name}")

    stats_name = f"ddqn2_frontier_stats_{BOARD_SIZE}x{BOARD_SIZE}_{NUM_MINES}m.npz"
    np.savez(
        stats_name,
        rewards=np.array(all_rewards, dtype=np.float32),
        steps=np.array(all_steps, dtype=np.int32),
        open_cells=np.array(all_open_cells, dtype=np.int32),
        progress=np.array(all_progress_counts, dtype=np.int32),
        guess=np.array(all_guess_counts, dtype=np.int32),
    )
    print(f"Training statistics have been saved as {stats_name}")


if __name__ == "__main__":
    main()