import numpy as np
import torch
from text_based_env import TextBasedEnv
from dqn_agent import DQNAgent

def train_dqn(agent, env, episodes=2000, max_steps=1000):
    best_avg_reward = -np.inf
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.learn()
            if done:
                break

        agent.update_epsilon()

        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    state_size = 1
    action_size = 4
    env = TextBasedEnv()
    agent = DQNAgent(state_size, action_size)
    train_dqn(agent, env)
