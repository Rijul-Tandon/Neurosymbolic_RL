# Import necessary libraries
# os: for interacting with the operating system, like getting file paths
import os
# random: for generating random numbers, used for seeding
import random
# time: for getting the current time, used for creating unique run names
import time
# dataclasses: for creating classes to store data, like our arguments
from dataclasses import dataclass

# Import libraries for reinforcement learning and neural networks
# gymnasium: A standard library for reinforcement learning environments. It provides the "game" for the agent to play and learn in.
import gymnasium as gym
# numpy: a library for numerical operations, especially with arrays
import numpy as np
# torch: the main PyTorch library for building and training neural networks
import torch
import torch.nn as nn  # nn: a module in PyTorch for building neural network layers
import torch.optim as optim  # optim: a module for optimization algorithms like Adam
# tyro: a library for easily creating command-line interfaces from Python classes
import tyro
# Categorical: a distribution from PyTorch, used for when actions are discrete (e.g., left or right)
from torch.distributions.categorical import Categorical
# SummaryWriter: for logging data to TensorBoard, which helps visualize training progress
from torch.utils.tensorboard import SummaryWriter


# A class to hold all the settings (hyperparameters) for our experiment
@dataclass
class Args:
    # The name of the experiment, taken from the filename by default
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    # The random seed, to make sure our results are reproducible
    seed: int = 1
    """seed of the experiment"""
    # Whether to use deterministic algorithms in PyTorch for reproducibility
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    # Whether to use a GPU (if available) for faster training
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    # Whether to track the experiment using Weights and Biases (a tool for experiment tracking)
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    # The project name in Weights and Biases
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    # The team name in Weights and Biases
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    # Whether to save videos of the agent playing the game
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # --- Algorithm specific arguments ---
    # The ID of the game environment from Gymnasium
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    # The total number of timesteps (actions taken) the agent will train for
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    # The learning rate for the optimizer, which controls how much the agent's brain (network) is updated each time
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    # The number of parallel game environments to run at the same time for faster data collection
    num_envs: int = 4
    """the number of parallel game environments"""
    # The number of steps to run in each parallel environment before updating the policy
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    # Whether to decrease the learning rate over time, which can help stabilize training
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    # The discount factor for future rewards. A value closer to 1 (like 0.99) means the agent is "far-sighted"
    # and cares a lot about rewards it will get in the future. A value of 0 would make it "short-sighted," only caring about the immediate reward.
    gamma: float = 0.99
    """the discount factor gamma"""
    # The lambda for Generalized Advantage Estimation (GAE). Advantage is a measure of how much better a specific action was
    # compared to the average action in that state. It helps the agent learn which actions are surprisingly good.
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    # The number of mini-batches to split the data into for each update.
    num_minibatches: int = 4
    """the number of mini-batches"""
    # The number of times to go over the collected data when updating the policy
    update_epochs: int = 4
    """the K epochs to update the policy"""
    # Whether to normalize the advantages, which can help stabilize training
    norm_adv: bool = True
    """Toggles advantages normalization"""
    # The clipping coefficient for the PPO algorithm. PPO works by making small, incremental updates to the policy.
    # This coefficient ensures that the policy doesn't change too drastically at once, which leads to more stable training.
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    # Whether to use a clipped loss for the value function, similar to the policy loss
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    # The coefficient for the entropy part of the loss. Entropy is a measure of randomness in the agent's policy.
    # A small entropy bonus encourages the agent to explore by trying different actions, preventing it from getting stuck in a rut.
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    # The coefficient for the value function part of the loss, which balances the policy and value updates
    vf_coef: float = 0.5
    """coefficient of the value function"""
    # The maximum value for the gradient norm, which prevents the gradients from getting too large and causing instability
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    # The target KL divergence, an alternative way to stop updates if the policy changes too much
    target_kl: float = None
    """the target KL divergence threshold"""

    # --- To be filled in at runtime ---
    # The total number of samples in one batch of data (num_envs * num_steps)
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    # The size of each mini-batch (batch_size / num_minibatches)
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    # The total number of training iterations (total_timesteps / batch_size)
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# A function to create a single game environment
def make_env(env_id, idx, capture_video, run_name):
    # A "thunk" is a function that is created now but called later. This is needed for creating parallel environments.
    def thunk():
        # If video capture is enabled and this is the first environment, create an environment that records video
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # Otherwise, create a standard environment
        else:
            env = gym.make(env_id)
        # Add a wrapper to automatically record statistics about each episode (like total reward and length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# A function to initialize the weights of a neural network layer
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Initialize the weights with an orthogonal distribution, which is a good practice for some networks
    torch.nn.init.orthogonal_(layer.weight, std)
    # Initialize the biases with a constant value
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# The Agent class defines the neural network architecture (the "brain")
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # The "critic" network is like a coach or a judge. It evaluates the current situation (state) of the game
        # and estimates a "value" for it. A high value means the state is promising and likely to lead to high future rewards.
        self.critic = nn.Sequential(
            # A linear layer takes the observation (state) as input and produces 64 features
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            # A Tanh activation function introduces non-linearity
            nn.Tanh(),
            # Another linear layer
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # The final layer outputs a single value, the estimated state value
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # The "actor" network is the player. It decides which action to take in a given state. It outputs a "policy,"
        # which is a strategy for choosing actions (e.g., push the cart left or right).
        self.actor = nn.Sequential(
            # A linear layer takes the observation (state) as input and produces 64 features
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            # Another linear layer
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # The final layer outputs a value for each possible action. These are called "logits".
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    # A method to get the value of a state from the critic network
    def get_value(self, x):
        return self.critic(x)

    # A method to get an action to take and the value of the state
    def get_action_and_value(self, x, action=None):
        # Get the action logits from the actor network
        logits = self.actor(x)
        # Create a probability distribution over the actions from the logits
        probs = Categorical(logits=logits)
        # If no action is provided, sample a new action from the distribution
        if action is None:
            action = probs.sample()
        # Return the chosen action, its log probability, the entropy of the distribution, and the state value from the critic
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# This is the main part of the script that runs when you execute the file
if __name__ == "__main__":
    # Parse the command-line arguments using the Args class
    args = tyro.cli(Args)
    # Calculate the batch size and minibatch size based on the arguments
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # Calculate the total number of iterations for the training loop
    args.num_iterations = args.total_timesteps // args.batch_size
    # Create a unique name for this run, including the environment, experiment name, seed, and current time
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # If tracking is enabled, initialize Weights and Biases
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,  # Sync with TensorBoard
            config=vars(args),  # Log all the arguments
            name=run_name,  # Set the run name
            monitor_gym=True,  # Automatically track gym environments
            save_code=True,  # Save the code for this run
        )
    # Create a SummaryWriter to log data to TensorBoard
    writer = SummaryWriter(f"runs/{run_name}")
    # Add the hyperparameters to the TensorBoard log as text
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # --- Seeding for reproducibility ---
    # Set the random seed for the `random` library
    random.seed(args.seed)
    # Set the random seed for NumPy
    np.random.seed(args.seed)
    # Set the random seed for PyTorch
    torch.manual_seed(args.seed)
    # Use deterministic algorithms in PyTorch's cuDNN backend
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # --- Device setup ---
    # Check if a CUDA-enabled GPU is available and if the user wants to use it
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- Environment setup ---
    # Create a vector of synchronous environments (they run in parallel)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    # Make sure the action space is discrete (e.g., a fixed number of choices)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # --- Agent and Optimizer setup ---
    # Create an instance of our Agent and move it to the selected device (CPU or GPU)
    agent = Agent(envs).to(device)
    # Create the Adam optimizer to update the agent's network parameters
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Storage setup for experience replay ---
    # Create tensors to store the data collected from the environments
    # Observations (what the agent sees)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # Actions (what the agent does)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # Log probabilities of the actions
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # Rewards (feedback from the environment)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # Dones (whether the episode has ended)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # Values (the critic's estimate of the state value)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # --- Main training loop ---
    # Initialize the global step counter
    global_step = 0
    # Record the start time to calculate steps per second (SPS)
    start_time = time.time()
    # Reset the environments to get the initial observation
    next_obs, _ = envs.reset(seed=args.seed)
    # Convert the observation to a PyTorch tensor and move it to the device
    next_obs = torch.Tensor(next_obs).to(device)
    # Initialize the 'done' flag for the next step
    next_done = torch.zeros(args.num_envs).to(device)

    # Loop for the total number of iterations
    for iteration in range(1, args.num_iterations + 1):
        # --- Learning rate annealing ---
        # If enabled, decrease the learning rate linearly over the course of training
        if args.anneal_lr:
            # Calculate the fraction of training remaining
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            # Calculate the new learning rate
            lrnow = frac * args.learning_rate
            # Set the optimizer's learning rate to the new value
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Data collection loop ---
        # Loop for the number of steps to run in each environment
        for step in range(0, args.num_steps):
            # Increment the global step counter
            global_step += args.num_envs
            # Store the current observation and done status
            obs[step] = next_obs
            dones[step] = next_done

            # --- Action selection ---
            # Temporarily disable gradient calculations, as we are just collecting data, not training
            with torch.no_grad():
                # Get an action, its log probability, and the state value from the agent
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # Store the value and action
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # --- Environment step ---
            # Execute the chosen action in the parallel environments
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # An episode is 'done' if it was terminated (e.g., cart fell) or truncated (e.g., time limit reached)
            next_done = np.logical_or(terminations, truncations)
            # Store the reward
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # Convert the next observation and done status to tensors
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # --- Logging episode information ---
            # Check if any episodes have finished
            if "final_info" in infos:
                # Loop through the info dictionaries for each environment
                for info in infos["final_info"]:
                    # If an info dict exists and contains episode data, it means an episode ended
                    if info and "episode" in info:
                        # Print the episodic return (total reward)
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        # Log the episodic return and length to TensorBoard
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # --- Advantage and Return Calculation (GAE) ---
        # After collecting a rollout, we calculate the advantages and returns
        with torch.no_grad():
            # Get the value of the last state to "bootstrap" from
            next_value = agent.get_value(next_obs).reshape(1, -1)
            # Initialize the advantages tensor
            advantages = torch.zeros_like(rewards).to(device)
            # Initialize the last GAE lambda value
            lastgaelam = 0
            # Loop backwards through the collected steps
            for t in reversed(range(args.num_steps)):
                # For the very last step, the next state is the one we just observed
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                # For all other steps, the next state is the one from the stored data
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # Calculate the TD error (delta): how much better the reward + next state's value was than expected
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # Calculate the advantage using GAE formula
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            # Calculate the returns by adding the advantages to the values
            returns = advantages + values

        # --- Flatten the batch for training ---
        # Reshape the collected data into a single long list (a "batch")
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # --- Policy and Value Network Optimization ---
        # Create an array of indices for the batch
        b_inds = np.arange(args.batch_size)
        # A list to store the fraction of samples that were clipped
        clipfracs = []
        # Loop for the number of update epochs
        for epoch in range(args.update_epochs):
            # Shuffle the indices to randomize the order of mini-batches
            np.random.shuffle(b_inds)
            # Loop through the batch in mini-batch chunks
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                # Get the indices for this mini-batch
                mb_inds = b_inds[start:end]

                # --- Calculate the loss for this mini-batch ---
                # Get the new log probabilities, entropy, and values for the observations in the mini-batch
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                # Calculate the ratio of the new policy to the old policy
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # --- Calculate approximate KL divergence (for logging) ---
                with torch.no_grad():
                    # A simple way to estimate KL divergence
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # Calculate the fraction of samples where the policy ratio was clipped
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Get the advantages for this mini-batch
                mb_advantages = b_advantages[mb_inds]
                # If enabled, normalize the advantages (subtract mean, divide by std dev)
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # --- Policy Loss (PPO's clipped objective) ---
                # The standard policy gradient loss
                pg_loss1 = -mb_advantages * ratio
                # The clipped policy gradient loss, which restricts the size of the policy update
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                # The final policy loss is the maximum of the two (takes the pessimistic bound)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- Value Loss ---
                newvalue = newvalue.view(-1)
                # If value clipping is enabled
                if args.clip_vloss:
                    # The standard squared error loss
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    # A clipped version of the value prediction
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    # The squared error loss with the clipped value
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    # Take the maximum of the unclipped and clipped losses
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    # The final value loss is half of the mean of this maximum
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # If not clipping, just use the standard mean squared error loss
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # --- Entropy Loss ---
                # The entropy loss encourages exploration by making the policy less deterministic
                entropy_loss = entropy.mean()
                # The total loss is a combination of policy loss, value loss, and entropy loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # --- Gradient Update ---
                # Reset the gradients of the optimizer
                optimizer.zero_grad()
                # Calculate the gradients of the loss with respect to the network parameters (backpropagation)
                loss.backward()
                # Clip the gradients to prevent them from becoming too large
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # Update the network parameters using the optimizer
                optimizer.step()

            # --- Early stopping (optional) ---
            # If the KL divergence exceeds the target, stop updating for this iteration
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # --- Calculate explained variance (for logging) ---
        # Get the predicted values and the actual returns
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # Calculate the variance of the returns
        var_y = np.var(y_true)
        # Calculate the explained variance, which measures how well the value function predicts the returns
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # --- Logging training metrics ---
        # Log the learning rate
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # Log the different components of the loss
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # Log the KL divergence metrics
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # Log the clipping fraction and explained variance
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # Print and log the steps per second (SPS)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # --- Cleanup ---
    # Close the environments
    envs.close()
    # Close the TensorBoard writer
    writer.close()
