"""
Training script for SelfAI project.
This script trains the DQN agent using the custom TextEnv environment.
"""

import numpy as np
import torch
from text_env5 import TextEnv
from dqn_agent import DoubleDQNAgent
from replay_buffer import PrioritizedReplayBuffer
import logging
import openai
import gym

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_dqn(agent, env, episodes=2000, max_steps=1000):
    best_avg_reward = -np.inf
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Log episode, step, action, reward, and target word
            logging.info(f"Episode: {episode}, Step: {step}, Action: {action}, Reward: {reward}, Target Word: {env.current_target}")

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.learn()

            if done:
                break

        agent.update_epsilon()

        if episode % 10 == 0:
            avg_reward = total_reward / max_steps
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            logging.info(f"Episode: {episode}, Total Reward: {total_reward}, Avg Reward: {avg_reward}, Best Avg Reward: {best_avg_reward}")

# Create an environment
env = TextEnv()

# Set your OpenAI API key
openai.api_key = env.openai_api_key

# Create a larger training dataset
training_dataset = generate_training_data(env, episodes=10000, max_steps=1000)

# Create a Gym environment using the training dataset
gym_env = gym.make('CustomEnv-v0', training_dataset=training_dataset)

# Pre-training
pretrain_episodes = 1000
pretrain_max_steps = 1000
pretrain_agent = DoubleDQNAgent(env.observation_space.shape[0], env.action_space.n)
pretrain_buffer = PrioritizedReplayBuffer()  # Use PrioritizedReplayBuffer
train_dqn(pretrain_agent, gym_env, episodes=pretrain_episodes, max_steps=pretrain_max_steps)

# Fine-tuning
fine_tune_episodes = 2000
fine_tune_max_steps = 1000
fine_tune_agent = DoubleDQNAgent(env.observation_space.shape[0], env.action_space.n)
fine_tune_buffer = PrioritizedReplayBuffer()  # Use PrioritizedReplayBuffer
fine_tune_agent.q_network.load_state_dict(pretrain_agent.q_network.state_dict())
train_dqn(fine_tune_agent, gym_env, episodes=fine_tune_episodes, max_steps=fine_tune_max_steps)
