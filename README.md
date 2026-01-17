# 🚀 Autonomous Lunar Lander (Double DQN)

This project implements a Deep Reinforcement Learning agent capable of autonomously landing a spacecraft in the LunarLander-v3 environment using Double Deep Q-Networks.

## 🧠 Key Concepts
- Deep Q-Network (DQN)
- Double DQN (reduces Q-value overestimation)
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration

## 🧪 Environment
- LunarLander-v3 (Gymnasium)
- Discrete action space (main engine, left/right engines)

## 📈 Results
- Average evaluation reward: **400+**
- Stable landing behavior
- Consistent policy performance

## 🎥 Demo

## 🛠️ How to Run
```bash
python -m venv rl-env
```
### windows
```bash
rl-env\Scripts\activate
```

### linux/mac
```bash
source rl-env/bin/activate
```

```bash
pip install gymnasium[box2d] torch
python dqn_lunarlander.py
python eval_lunarlander.py
