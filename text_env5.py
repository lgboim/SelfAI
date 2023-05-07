import gym
from gym import spaces
import openai
import random
import logging
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import time

# Load pre-trained word vectors
word_vectors = api.load("glove-wiki-gigaword-50")

# Set up logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class TextEnv(gym.Env):
    def __init__(self):
        super(TextEnv, self).__init__()
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(100,), dtype=int)
        self.openai_api_key = "openai-key"
        self.target_words = ["apple", "banana", "cherry", "date", "fig"]
        self.current_target = random.choice(self.target_words)
        self.previous_state = None  # Track the previous state

    def step(self, action):
        # Generate text based on the action
        text = self.generate_text(action)

        # Compute the reward based on the generated text
        reward = self.compute_reward(text)

        # Create an observation
        observation = self.create_observation()

        done = False  # Change this to True when the environment terminates
        info = {}  # Additional information, if needed

        return observation, reward, done, info

    def reset(self, start_from_same_state=False):
        if start_from_same_state and self.previous_state is not None:
            return self.previous_state  # Start from the previous state

        self.previous_state = self.create_observation()
        return self.previous_state

    def render(self, mode='human'):
        pass

    def generate_text(self, action):
        input_text = "I am a large language model."
        prompt = f"{input_text} Action: {action}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            n=1,
            stop=None,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        generated_text = response.choices[0].text.strip()
        return generated_text

    def extract_words(self, text):
        words = text.lower().split()
        words = [word.strip('.,!?:;()[]') for word in words]
        return words

    def calculate_word_similarity_score(self, matching_words):
        num_matching_words = len(matching_words)
        num_target_words = len(self.target_words)
        
        if num_target_words == 0:
            return 0.0
        
        return num_matching_words / num_target_words

    def compute_reward(self, text):
        words = self.extract_words(text)
        matching_words = [word for word in words if word in self.target_words]

        # Calculate the word similarity score between the generated text and the target words.
        word_similarity_score = self.calculate_word_similarity_score(matching_words)

        # Calculate the length penalty.
        length_penalty = self.calculate_length_penalty(len(words))

        # Calculate the fluency score.
        fluency_score = self.calculate_fluency_score(text)

        # Calculate the accuracy score.
        accuracy_score = self.calculate_accuracy_score(text)

        # Calculate the total reward.
        reward = word_similarity_score * 0.5 + length_penalty * 0.2 + fluency_score * 0.2 + accuracy_score * 0.1

        return reward

    def calculate_length_penalty(self, text_length):
        max_length = 50
        penalty_factor = 0.02
        if text_length > max_length:
            return 1.0 - (text_length - max_length) * penalty_factor
        return 1.0

    def calculate_fluency_score(self, text):
        # Your fluency score calculation logic goes here
        return 0.9

    def calculate_accuracy_score(self, text):
        # Your accuracy score calculation logic goes here
        return 0.8

    def create_observation(self):
        # Implement your observation creation logic here
        # You can represent the observation as a vector of relevant features
        observation = [0] * 100  # Replace with your own observation representation
        return observation

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.001  # Learning rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.memory = deque(maxlen=2000)

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size,), activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def learn(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            self.model.fit(state, target, epochs=1, verbose=0)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create an environment
env = TextEnv()

# Set your OpenAI API key
openai.api_key = env.openai_api_key

# Set random seed for reproducibility
np.random.seed(0)
random.seed(0)

# Set up the DQN agent
state_size = len(env.create_observation())
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training parameters
episodes = 2000
max_steps = 1000
update_target_every = 10
early_stopping_patience = 10

# Training loop
agent.epsilon = 1.0
best_total_reward = -np.inf
stagnation_counter = 0

for episode in range(episodes):
    state = env.reset(start_from_same_state=True)
    total_reward = 0
    for step in range(max_steps):
        action = agent.act(np.array([state]))
        next_state, reward, done, _ = env.step(action)

        # Log episode, step, action, reward, and target word
        logging.info(f"Episode: {episode}, Step: {step}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")

        agent.remember(np.array([state]), action, reward, np.array([next_state]), done)
        state = next_state
        total_reward += reward
        agent.learn()
        agent.update_epsilon()
        time.sleep(0.1)  # Add a short delay of 0.1 seconds between steps

        if done:
            break

    if episode % update_target_every == 0:
        agent.update_target_model()

    if total_reward > best_total_reward:
        best_total_reward = total_reward
        stagnation_counter = 0
    else:
        stagnation_counter += 1
        if stagnation_counter >= early_stopping_patience:
            break

    if episode % 10 == 0:
        logging.info(f"Episode: {episode}, Total Reward: {total_reward}")

# Add methods to save/load the model, if needed
