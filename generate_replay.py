import gymnasium as gym
import torch
import json
import torch.nn as nn
import torch.nn.functional as F

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

env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("models/double_dqn_lunarlander.pth"))
model.eval()

state, _ = env.reset()
episode = []

for t in range(1000):
    with torch.no_grad():
        q = model(torch.FloatTensor(state))
        action = torch.argmax(q).item()

    next_state, reward, terminated, truncated, _ = env.step(action)

    episode.append({
        "t": t,
        "state": {
            "x": float(state[0]),
            "y": float(state[1]),
            "vx": float(state[2]),
            "vy": float(state[3]),
            "angle": float(state[4]),
            "angular_velocity": float(state[5]),
            "left_leg": int(state[6]),
            "right_leg": int(state[7]),
        },
        "action": action,
        "reward": float(reward)
    })

    state = next_state
    if terminated or truncated:
        break

with open("episode.json", "w") as f:
    json.dump(episode, f, indent=2)

env.close()
print("Replay saved!")
