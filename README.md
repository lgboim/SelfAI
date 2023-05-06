# DQN Agent

A Deep Q-Network (DQN) agent implementation that learns to interact with a text-based environment using reinforcement learning. Built with PyTorch, this project aims to provide an easy-to-use DQN agent that can be customized and extended for various use cases. We'd love experienced developers to contribute and help improve this initial experience!

## Features

- DQN neural network architecture for approximating the Q-function.
- Epsilon-greedy policy for balancing exploration and exploitation.
- Experience replay to store and learn from past experiences.
- Separate target network for stable training.
- Example script to demonstrate usage with a custom text-based environment.

## Dependencies

To get started with the DQN agent, ensure you have the following Python libraries installed:

- NumPy
- PyTorch
- gym
- openai

## File Structure

- `text_env5.py`: Contains the `TextEnv` class, a custom text-based environment that inherits from the `gym.Env` class.
- `train.py`: The main script to train the DQN agent within the text-based environment.
- `dqn_agent.py`: Defines the DQN neural network architecture and DQNAgent class, which manages agent actions, learning, and memory.
- `observations.py`: Contains the `create_observation` function for processing observations from the environment.

## Getting Started

1. Clone the repository to your local machine.
2. Install the required dependencies (NumPy, PyTorch, gym, openai) if not already installed. You can use `pip` to install them:

pip install numpy torch gym openai

3. Set up your OpenAI API key as an environment variable named OPENAI_API_KEY. Replace `"YOUR_API_KEY"` with your actual OpenAI API key:
- Windows:
  ```
  set OPENAI_API_KEY=YOUR_API_KEY
  ```
- Linux/Mac:
  ```
  export OPENAI_API_KEY=YOUR_API_KEY
  ```
4. Run the `train.py` script to train the DQN agent and observe its progress. You can execute the script using the following command:

python train.py

## Contributing

We'd love to have experienced developers join our journey in creating a versatile and powerful DQN agent. If you're interested in contributing, please check the open issues, submit new issues or feature requests, and submit pull requests with improvements and bug fixes.

