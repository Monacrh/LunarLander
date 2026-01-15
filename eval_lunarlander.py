# import gymnasium as gym
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # =========================
# # Q-Network (harus sama)
# # =========================
# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, action_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


# def evaluate():
#     env = gym.make("LunarLander-v3", render_mode="human")

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n

#     model = DQN(state_dim, action_dim)
#     model.load_state_dict(torch.load("models/double_dqn_lander.pth"))
#     model.eval()

#     state, _ = env.reset()
#     total_reward = 0
#     done = False

#     while not done:
#         with torch.no_grad():
#             q_values = model(torch.FloatTensor(state))
#             action = torch.argmax(q_values).item()

#         state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         total_reward += reward

#     print(f"Total Reward: {total_reward:.2f}")
#     env.close()

# if __name__ == "__main__":
#     evaluate()
 
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def evaluate(episodes=5, render=True):
    env = gym.make(
        "LunarLander-v3",
        render_mode="human" if render else None
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load("models/double_dqn_lunarlander.pth"))
    model.eval()

    scores = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action = torch.argmax(model(state_tensor)).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        scores.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.1f}")

    print(f"\nAverage Reward: {sum(scores)/len(scores):.1f}")
    env.close()


if __name__ == "__main__":
    evaluate()
