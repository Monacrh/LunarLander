import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

# =========================
# Q-Network
# =========================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.LongTensor(a).unsqueeze(1),
            torch.FloatTensor(r),
            torch.FloatTensor(s_next),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buffer)

# =========================
# Training
# =========================
def train():
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    batch_size = 64
    target_update = 1000
    max_episodes = 1000

    step_count = 0

    for ep in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            step_count += 1

            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(state))
                    action = torch.argmax(q).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train
            if len(buffer) >= batch_size:
                s, a, r, s_next, d = buffer.sample(batch_size)

                q_values = policy_net(s).gather(1, a).squeeze()
                with torch.no_grad():
                    next_actions = policy_net(s_next).argmax(1)
                    next_q = target_net(s_next).gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze()

                target = r + gamma * next_q * (1 - d)
                loss = F.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if step_count % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

        # Early stop (LunarLander solved ~200)
        if total_reward >= 200:
            print("Environment solved!")
            break

    torch.save(policy_net.state_dict(), "models/double_dqn_lander.pth")
    env.close()


if __name__ == "__main__":
    train()
