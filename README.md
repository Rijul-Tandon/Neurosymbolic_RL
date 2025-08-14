# Neurosymbolic Reinforcement Learning: Bipedal Walker Policy

A sophisticated reinforcement learning implementation using Proximal Policy Optimization (PPO) for training bipedal walker agents. This project combines neural network-based policy learning with comprehensive logging and evaluation capabilities.

## 🤖 Project Overview

This project implements a state-of-the-art PPO algorithm designed to train autonomous bipedal walking agents. The implementation features:

- **Proximal Policy Optimization (PPO)**: A stable and efficient policy gradient method
- **Actor-Critic Architecture**: Separate neural networks for policy (actor) and value estimation (critic)
- **Parallel Environment Training**: Multi-environment simulation for faster data collection
- **Comprehensive Logging**: TensorBoard integration for training visualization
- **Flexible Environment Support**: Compatible with various Gymnasium environments

## 🚀 Features

### Core Algorithm Features
- **PPO Implementation**: State-of-the-art policy optimization with clipping for stable updates
- **Generalized Advantage Estimation (GAE)**: Advanced advantage calculation for improved learning
- **Learning Rate Annealing**: Adaptive learning rate scheduling
- **Gradient Clipping**: Prevents training instability from large gradients
- **Entropy Regularization**: Encourages exploration during training

### Technical Features
- **Multi-Environment Parallel Training**: Run multiple environments simultaneously
- **Video Recording Capability**: Capture agent performance videos
- **Comprehensive Metrics Logging**: Track training progress with detailed analytics
- **Reproducible Results**: Seeded random number generation for consistent experiments
- **GPU Acceleration**: CUDA support for faster training

## 📋 Requirements

```bash
# Core dependencies
gymnasium>=0.26.0
torch>=1.12.0
numpy>=1.21.0
tensorboard>=2.8.0
tyro>=0.3.0

# Optional dependencies for specific environments
gymnasium[box2d]  # For BipedalWalker environments
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rijul-Tandon/Neurosymbolic_RL.git
   cd Neurosymbolic_RL
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install box2d  # For bipedal walker environments
   ```

## 🎮 Usage

### Basic Training

Train a bipedal walker agent with default settings:
```bash
python ppo_continuous_action_original.py --env-id BipedalWalker-v3
```

### Advanced Configuration

Train with custom hyperparameters:
```bash
python ppo_continuous_action_original.py \
    --env-id BipedalWalker-v3 \
    --total-timesteps 2000000 \
    --learning-rate 3e-4 \
    --num-envs 8 \
    --num-steps 2048 \
    --capture-video
```

### Available Environments

The implementation supports various bipedal walker environments:
- `BipedalWalker-v3`: Standard bipedal walking task
- `BipedalWalkerHardcore-v3`: Challenging terrain navigation
- `CartPole-v1`: Simple control task (default for testing)

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--env-id` | Environment identifier | `CartPole-v1` |
| `--total-timesteps` | Total training steps | `500000` |
| `--learning-rate` | Optimizer learning rate | `2.5e-4` |
| `--num-envs` | Parallel environments | `4` |
| `--num-steps` | Steps per environment rollout | `128` |
| `--capture-video` | Record training videos | `False` |
| `--track` | Enable Weights & Biases tracking | `False` |

## 📊 Monitoring Training

### TensorBoard Visualization

Monitor training progress in real-time:
```bash
tensorboard --logdir runs
```

### Tracked Metrics

- **Episode Rewards**: Cumulative reward per episode
- **Episode Length**: Number of steps per episode
- **Policy Loss**: Actor network optimization loss
- **Value Loss**: Critic network optimization loss
- **Entropy**: Policy exploration measure
- **KL Divergence**: Policy change magnitude
- **Learning Rate**: Current optimization rate

## 🏗️ Architecture

### Neural Network Design

**Actor Network (Policy)**:
- Input: Environment observations
- Hidden: 2 layers × 64 neurons (Tanh activation)
- Output: Action logits for policy distribution

**Critic Network (Value Function)**:
- Input: Environment observations  
- Hidden: 2 layers × 64 neurons (Tanh activation)
- Output: State value estimation

### Algorithm Components

1. **Experience Collection**: Gather trajectories from parallel environments
2. **Advantage Estimation**: Calculate advantages using GAE
3. **Policy Updates**: Optimize policy with PPO clipping objective
4. **Value Updates**: Train critic to predict state values
5. **Entropy Regularization**: Maintain exploration capabilities

## 🎯 Performance

### Expected Results

For BipedalWalker-v3:
- **Training Time**: ~2-4 hours (depending on hardware)
- **Target Reward**: 300+ (successful walking)
- **Sample Efficiency**: Convergence within 1-2M timesteps

### Optimization Tips

1. **Increase parallel environments** (`--num-envs 16`) for faster data collection
2. **Adjust learning rate** (`--learning-rate 1e-4`) for stability
3. **Extend training** (`--total-timesteps 2000000`) for complex environments
4. **Enable video capture** (`--capture-video`) to visualize progress

## 🔬 Neurosymbolic Aspects

This implementation serves as a foundation for neurosymbolic reinforcement learning research:

- **Neural Component**: Deep neural networks for perception and decision-making
- **Symbolic Integration**: Structured reward engineering and interpretable policies
- **Hybrid Learning**: Combination of data-driven and knowledge-based approaches

## 📁 Project Structure

```
Neurosymbolic_RL/
├── ppo.py              # Main PPO implementation
├── README.md           # Project documentation
├── runs/               # TensorBoard logs (generated)
└── videos/             # Training videos (if enabled)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional environment support
- Advanced neurosymbolic integration
- Hyperparameter optimization
- Multi-agent scenarios
- Curriculum learning

## 📚 References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Rijul Tandon**
- GitHub: [@Rijul-Tandon](https://github.com/Rijul-Tandon)

---

**Happy Training! 🤖🚶‍♂️**



