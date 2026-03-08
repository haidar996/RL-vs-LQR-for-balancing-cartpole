# RL vs LQR for Cart-Pole Balancing (ROS + Gazebo)

[![ROS](https://img.shields.io/badge/ROS-Noetic-blue)](http://wiki.ros.org/noetic)
[![Gazebo](https://img.shields.io/badge/Gazebo-9+-orange)](http://gazebosim.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red)](https://pytorch.org/)

A comparative study of **classical optimal control** (LQR) versus **deep reinforcement learning** (DDQN) for stabilizing an inverted pendulum system in a realistic ROS + Gazebo simulation environment.

---

## 📋 Overview

This project investigates two fundamentally different approaches for controlling the classical inverted pendulum (Cart-Pole) system:

- **Model-based optimal control** using Linear Quadratic Regulator (LQR)
- **Model-free learning** using Double Deep Q-Network (DDQN)

The system is implemented and tested in the robotics framework **ROS** with **Gazebo** simulation. The goal is to compare classical control theory vs reinforcement learning for stabilizing the inverted pendulum while observing:

- System response
- Control effort
- Learning convergence
- Robustness to different setpoints

---

## 🎯 The Cart-Pole System

The Cart-Pole is one of the most famous benchmark control problems used in both control theory and reinforcement learning research. The system consists of a cart moving on a rail with a pendulum attached at its center.

### State Vector

The system state is represented as:
x = [x, ẋ, θ, θ̇]

| Variable | Description |
|----------|-------------|
| x | Cart position |
| ẋ | Cart velocity |
| θ | Pole angle |
| θ̇ | Pole angular velocity |

The control input is `u = F` (horizontal force applied to the cart).

---

## 🏗️ Project Structure
RL-vs-LQR-for-balancing-cartpole/
├── src/
│ └── cart_pole/
│ └── src/
│ ├── robot_description/ # URDF models and meshes
│ ├── robot_control/ # ROS controller configurations
│ ├── robot_launch/ # Launch files for simulation
│ └── commander/
│ └── scripts/ # Control algorithms
│ ├── lqr.py # LQR controller
│ ├── DDQN.py # Neural network implementation
│ ├── DDQNAGENT.py # DDQN agent logic
│ └── train_ddqn.py # RL training loop
├── LQR/ # LQR experiment results
│ ├── zeros_setpoints/
│ ├── theta_Setpoint/
│ ├── setpointx/
│ ├── no_controller/
│ └── xchanged/
├── RL/ # RL experiment results
├── videos/ # Simulation recordings
└── README.md

---

## 1️⃣ LQR Controller

### Concept

The Linear Quadratic Regulator (LQR) is an optimal state feedback controller designed from the **linearized state-space model**:
## 1️⃣ LQR Controller

### Concept

The Linear Quadratic Regulator (LQR) is an optimal state feedback controller designed from the **linearized state-space model**:
ẋ = Ax + Bu

The control law is `u = -Kx`, where the gain matrix `K` is computed by minimizing the quadratic cost function:


J = ∫(xᵀQx + uᵀRu)dt

| Matrix | Meaning |
|--------|---------|
| Q | State error penalty |
| R | Control effort penalty |

### Implementation

The controller is implemented in `src/cart_pole/src/commander/scripts/lqr.py`:

- Defines system matrices A, B
- Computes LQR gain matrix K
- Subscribes to robot states from ROS
- Computes control force using state feedback
- Publishes commands to the robot controller

### LQR Parameters

**State penalty matrix:**

| Matrix | Meaning |
|--------|---------|
| Q | State error penalty |
| R | Control effort penalty |

### Implementation

The controller is implemented in `src/cart_pole/src/commander/scripts/lqr.py`:

- Defines system matrices A, B
- Computes LQR gain matrix K
- Subscribes to robot states from ROS
- Computes control force using state feedback
- Publishes commands to the robot controller

### LQR Parameters

**State penalty matrix:**
Q = diag(10, 1, 100, 1)

**Control effort penalty:**
R = 0.01

These matrices prioritize keeping the pole upright while minimizing excessive control force.

### LQR Gain Matrix

The gain matrix K computed from the Riccati equation has the form:
K = [k₁, k₂, k₃, k₄]

which multiplies the system state vector to compute the control force.

### Experiments

The repository includes several LQR experiments under the `LQR/` directory demonstrating:
- Stabilization around equilibrium
- Response to non-zero setpoints
- Uncontrolled system dynamics

Each experiment includes ROS bag recordings, state plots, control effort plots, and simulation videos.

---

## 2️⃣ Reinforcement Learning Controller

### Algorithm: Double Deep Q-Network (DDQN)

The learning controller is based on **Double Deep Q-Network (DDQN)**, which improves the stability of the original DQN by separating:
- Action selection network
- Target Q-value network

Deep reinforcement learning algorithms learn control policies through interaction with the environment, **without requiring the system model**.

### Implementation Files
src/cart_pole/src/commander/scripts/
├── DDQN.py # Neural network architecture
├── DDQNAGENT.py # Agent logic (replay buffer, action selection, learning)
└── train_ddqn.py # Main training loop

The network is implemented using **PyTorch**.

### State Representation

The agent observes the same system state as the LQR controller:
s = [x, ẋ, θ, θ̇]

These four values form the input layer of the neural network.

### Action Space

The RL agent outputs discrete actions:

| Action | Meaning |
|--------|---------|
| 0 | Apply force left |
| 1 | Apply force right |

### Reward Design

The reward encourages the pole to stay upright while keeping the cart within bounds:

- +1 reward per timestep
- Penalty if pole angle exceeds threshold
- Penalty if cart position exceeds track limits

Episode terminates if:
- Pole angle exceeds threshold (typically ±12°)
- Cart reaches track limit

### Neural Network Architecture
Input layer : 4 neurons (state)
Hidden layer : 128 neurons (ReLU)
Hidden layer : 128 neurons (ReLU)
Output layer : 2 neurons (Q-values for each action)

The output layer produces `Q(s,0)` and `Q(s,1)` representing the expected return for each action.

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| learning_rate | 0.001 | Optimizer step size |
| gamma | 0.99 | Reward discount factor |
| epsilon_start | 1.0 | Initial exploration probability |
| epsilon_min | 0.01 | Minimum exploration probability |
| epsilon_decay | 0.995 | Decay factor for epsilon |
| batch_size | 64 | Training batch size |
| replay_buffer_size | 100000 | Experience replay capacity |
| target_update_frequency | 1000 steps | Target network update period |
| max_episodes | 10000 | Number of training episodes |

### Experience Replay

The agent stores transitions in replay memory:
(s, a, r, s', done)

where:
- `s` = current state
- `a` = chosen action
- `r` = received reward
- `s'` = next state
- `done` = terminal state flag

Random mini-batches are sampled during training to break correlations between samples.

### Training Loop
for episode in range(max_episodes):
reset environment
while not done:
choose action using ε-greedy policy
apply action to simulation
observe reward and next state
store transition in replay buffer
sample random batch
compute target Q-value
update neural network
periodically update target network

---

## 🔧 ROS Architecture

The system is organized as several ROS packages:

### robot_description
Contains URDF model, robot meshes, and Gazebo simulation description.

### robot_control
Contains `controller.yaml` for ROS controllers.

### robot_launch
Contains launch files to start:
- Gazebo simulation
- Controllers
- Robot model

### commander
Contains all control algorithms:
- LQR controller
- RL training
- RL evaluation

### Control Pipeline
Gazebo Simulation
↓
robot_state_publisher
↓
commander node (LQR/RL)
↓
controller command
↓
Cart Actuator

---

## 📊 Experimental Results

The repository includes:
- 📈 Plots of system states over time
- 🎥 Simulation videos
- 📉 Training curves (for RL)
- 📦 ROS bag recordings

### Key Performance Metrics Analyzed:
- Pole angle stability
- Cart position response
- Control effort
- Learning convergence (RL)
- Robustness to different setpoints

### Demonstration Videos
Simulation videos are provided in the `videos/` directory:
- LQR stabilization
- RL learned controller
- Uncontrolled system response

---

## 🚀 Getting Started

### Prerequisites
- ROS (Noetic recommended)
- Gazebo 9+
- Python 3
- PyTorch

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/haidar996/RL-vs-LQR-for-balancing-cartpole.git
   cd RL-vs-LQR-for-balancing-cartpole
   Build the workspace

bash
catkin_make
Source ROS workspace

bash
source devel/setup.bash
Launch the simulation

bash
roslaunch robot_launch launch_simulation.launch
Run LQR controller

bash
rosrun commander lqr.py
Run RL training

bash
python3 train_ddqn.py
🎯 Project Motivation
Classical optimal control methods like LQR require:

✅ Accurate system model

✅ Linearization around operating point

✅ Manual tuning

Reinforcement learning can:

✅ Learn control policies without explicit system modeling

✅ Adapt to nonlinear dynamics

❌ Requires large training data

❌ Needs careful hyperparameter tuning

This project demonstrates the differences between the two approaches on the same robotic system.

📈 Key Takeaways
Feature	LQR	RL
Requires model	✅ Yes	❌ No
Training required	❌ No	✅ Yes
Optimal near equilibrium	✅ Yes	⚠️ Sometimes
Adaptability	❌ Low	✅ High
Computational cost	✅ Low	❌ High
Interpretability	✅ High	❌ Low
🔮 Future Work
Continuous action RL algorithms (DDPG, SAC, PPO)

Domain randomization for robust policies

Sim-to-real transfer

Nonlinear model predictive control (MPC) comparison

Hardware implementation on physical cart-pole system

👨‍💻 Author
Haidar Saad
Aerospace Engineering — Robotics and Control Systems

https://img.shields.io/badge/GitHub-haidar996-blue

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Classical control theory foundations

Deep reinforcement learning research community

ROS and Gazebo open-source communities
