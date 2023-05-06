"""
Training script for SelfAI project.
This script trains the DQN agent using the custom TextEnv environment.
"""

import numpy as np
import torch
from text_env5 import TextEnv
from dqn_agent import DQNAgent
import logging
import openai

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

# Use the environment and train/test the agent as desired

if __name__ == "__main__":
    state_size = 1
    action_size = 4
    env = TextEnv()
    agent = DQNAgent(state_size, action_size)
    train_dqn(agent, env)
