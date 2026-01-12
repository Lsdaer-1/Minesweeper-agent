import numpy as np
import pygame
import torch
import imageio.v2 as imageio  # save gif

from minesweeper import MinesweeperDiscreetEnv
from agent import (
    MinesweeperDQN,
    preprocess_state,
    compute_frontier_mask,
    CLOSED_VAL,
    MINE_VAL,
    BOARD_SIZE,
    NUM_MINES,
    device,
)


MODEL_PATH = "ddqn2o_frontier_3_10x10_9m.pt"   
NUM_TEST_EPISODES = 5            
MAX_STEPS_PER_EPISODE = 150

TILE_SIZE = 32
MARGIN = 2
FPS = 1     

COLOR_BG        = (30, 30, 30)
COLOR_COVERED   = (120, 120, 120)
COLOR_UNCOVERED = (230, 230, 230)
COLOR_MINE      = (200, 40, 40)
COLOR_TEXT      = (0, 0, 0)
COLOR_FRONTIER  = (180, 180, 250)
COLOR_FLAG      = (255, 215, 0)    

SAVE_GIF = True
GIF_PATH = "minesweeper_demo.gif"
GIF_FPS = 1       


# ============ FLAG ============
def compute_flag_mask(raw_state: np.ndarray) -> np.ndarray:
    """
    Simple Flagging Rules (Not guaranteed to be 100% accurate, just for visual reference):
     For each uncovered number cell n:
     U = Uncovered cells in the neighborhood
     F = Cells in the neighborhood already marked with flags (as deduced in this round)
         If U + F == n → U contains only mines, place a flag
    Return flag_mask: (H,W) bool, where cells marked True are displayed as “F”.
    """
    H, W = raw_state.shape
    flag_mask = np.zeros_like(raw_state, dtype=bool)

    for i in range(H):
        for j in range(W):
            v = raw_state[i, j]
            if v < 0 or v > 8:
                continue

            number = int(v)

            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbors.append((ni, nj))

            U = [(ni, nj) for (ni, nj) in neighbors if raw_state[ni, nj] == CLOSED_VAL]
            F = [(ni, nj) for (ni, nj) in neighbors if flag_mask[ni, nj]]

            if len(U) == 0:
                continue

            if len(U) + len(F) == number:
                for (ni, nj) in U:
                    flag_mask[ni, nj] = True

    return flag_mask


# ============ frontier mask） ============
def get_action_mask_frontier_eval(env, raw_state: np.ndarray,
                                  num_actions: int,
                                  first_move: bool) -> np.ndarray:
    
    flat_closed = (raw_state.flatten() == CLOSED_VAL)

    if first_move:
        not_mine = (env.board.flatten() != MINE_VAL)
        return flat_closed & not_mine

    frontier = compute_frontier_mask(raw_state).flatten()
    if frontier.any():
        return frontier
    else:
        return flat_closed



def greedy_action_and_q(policy_net, state_np: np.ndarray,
                        num_actions: int,
                        valid_mask: np.ndarray):
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        dummy_q = np.zeros(num_actions, dtype=np.float32)
        return int(np.random.randint(0, num_actions)), dummy_q

    state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_tensor).squeeze(0).cpu().numpy()   # (num_actions,)

    q_masked = q_values.copy()
    q_masked[~valid_mask] = -1e9
    action = int(q_masked.argmax())
    return action, q_values


def q_to_color(q, q_min, q_max):
    if q_max <= q_min:
        t = 0.5
    else:
        t = (q - q_min) / (q_max - q_min)
        t = float(np.clip(t, 0.0, 1.0))

    r = int(255 * (1.0 - t))
    g = int(255 * t)
    b = 0
    return (r, g, b)



def draw_board(screen, font,
               raw_state: np.ndarray,
               frontier_mask: np.ndarray,
               flag_mask: np.ndarray,
               q_values: np.ndarray,
               valid_mask: np.ndarray):
    """
    raw_state : (H,W)
    q_values  : (num_actions,)
    valid_mask: (num_actions,) bool
    """
    screen.fill(COLOR_BG)
    H, W = raw_state.shape

    valid_q = q_values[valid_mask]
    if valid_q.size > 0:
        q_min = float(valid_q.min())
        q_max = float(valid_q.max())
    else:
        q_min, q_max = 0.0, 1.0

    for i in range(H):
        for j in range(W):
            idx = i * W + j
            x = j * (TILE_SIZE + MARGIN) + MARGIN
            y = i * (TILE_SIZE + MARGIN) + MARGIN
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

            v = int(raw_state[i, j])

            # 1) mine: Direct red background + X (overlays everything)
            if v == MINE_VAL:
                pygame.draw.rect(screen, COLOR_MINE, rect)
                text = font.render("X", True, (255, 255, 255))
                screen.blit(text, text.get_rect(center=rect.center))
                continue

            # 2) Unrevealed cells: Colored using Q-value heatmaps
            if v == CLOSED_VAL:
                if valid_mask[idx] and valid_q.size > 0:
                    color = q_to_color(q_values[idx], q_min, q_max)
                else:
                    if frontier_mask[i, j]:
                        color = COLOR_FRONTIER
                    else:
                        color = COLOR_COVERED
                pygame.draw.rect(screen, color, rect)

                # Flag: Overlay a golden F on the heatmap background
                if flag_mask[i, j]:
                    text = font.render("F", True, COLOR_FLAG)
                    screen.blit(text, text.get_rect(center=rect.center))

                continue

            pygame.draw.rect(screen, COLOR_UNCOVERED, rect)
            if 1 <= v <= 8:
                text = font.render(str(v), True, COLOR_TEXT)
                screen.blit(text, text.get_rect(center=rect.center))



def evaluate_pygame(model_path=MODEL_PATH):
    env = MinesweeperDiscreetEnv(board_size=BOARD_SIZE, num_mines=NUM_MINES)
    num_actions = env.action_space.n

    model = MinesweeperDQN(BOARD_SIZE, num_actions).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Using device: {device}, loaded model: {model_path}")

    # pygame ini
    pygame.init()
    font = pygame.font.SysFont("Arial", 18)

    width_px  = BOARD_SIZE * (TILE_SIZE + MARGIN) + MARGIN
    height_px = BOARD_SIZE * (TILE_SIZE + MARGIN) + MARGIN
    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("Minesweeper DDQN - Q Heatmap + Flags")

    clock = pygame.time.Clock()

    gif_frames = []

    for ep in range(1, NUM_TEST_EPISODES + 1):
        raw_state = env.reset()
        state = preprocess_state(raw_state)

        done = False
        steps = 0
        first_move = True
        running = True
        last_env_reward = 0

        print(f"\n=== Episode {ep} ===")

        frontier_mask = compute_frontier_mask(raw_state)
        flag_mask = compute_flag_mask(raw_state)
        valid_mask = get_action_mask_frontier_eval(
            env, raw_state, num_actions, first_move
        )
        _, q_values = greedy_action_and_q(model, state, num_actions, valid_mask)
        draw_board(screen, font, raw_state,
                   frontier_mask, flag_mask,
                   q_values, valid_mask)
        pygame.display.flip()

        if SAVE_GIF:
            arr = pygame.surfarray.array3d(screen)        # (W,H,3)
            arr = np.transpose(arr, (1, 0, 2))           # (H,W,3)
            gif_frames.append(arr.copy())

        while running and not done and steps < MAX_STEPS_PER_EPISODE:
            steps += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True

            frontier_mask = compute_frontier_mask(raw_state)
            flag_mask = compute_flag_mask(raw_state)

            # frontier 
            valid_mask = get_action_mask_frontier_eval(
                env, raw_state, num_actions, first_move
            )

            action, q_values = greedy_action_and_q(
                model, state, num_actions, valid_mask
            )

            draw_board(screen, font, raw_state,
                       frontier_mask, flag_mask,
                       q_values, valid_mask)
            pygame.display.flip()

            if SAVE_GIF:
                arr = pygame.surfarray.array3d(screen)
                arr = np.transpose(arr, (1, 0, 2))
                gif_frames.append(arr.copy())

            clock.tick(FPS)

            
            next_raw_state, env_reward, done, _ = env.step(action)
            last_env_reward = env_reward
            raw_state = next_raw_state
            state = preprocess_state(raw_state)
            first_move = False

        frontier_mask = compute_frontier_mask(raw_state)
        flag_mask = compute_flag_mask(raw_state)
        valid_mask = get_action_mask_frontier_eval(
            env, raw_state, num_actions, False
        )
        _, q_values = greedy_action_and_q(model, state, num_actions, valid_mask)
        draw_board(screen, font, raw_state,
                   frontier_mask, flag_mask,
                   q_values, valid_mask)
        pygame.display.flip()
        pygame.time.wait(600)

        if SAVE_GIF:
            arr = pygame.surfarray.array3d(screen)
            arr = np.transpose(arr, (1, 0, 2))
            gif_frames.append(arr.copy())

        open_cells = int((raw_state != CLOSED_VAL).sum())
        if last_env_reward == 1000:
            result_str = "WIN"
        elif last_env_reward == -100:
            result_str = "LOSE"
        else:
            result_str = "END"

        print(f"[Eval] Ep {ep}: steps={steps}, env_reward={last_env_reward}, "
              f"open_cells={open_cells}, result={result_str}")

        if not running:
            break

    env.close()
    pygame.quit()

    if SAVE_GIF and len(gif_frames) > 0:
        imageio.mimsave(GIF_PATH, gif_frames, fps=GIF_FPS)
        print(f"GIF save as: {GIF_PATH}")


if __name__ == "__main__":
    evaluate_pygame()