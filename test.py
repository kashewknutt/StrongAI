import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

env = gym.make("CartPole-v1")

# Hyperparameters
num_actions = env.action_space.n
state_shape = env.observation_space.shape
learning_rate = 0.001
gamma = 0.99  # Discount factor for future rewards
episodes = 1000  # Number of episodes to train
batch_size = 64  # Number of samples per batch for training

# Model
model = keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=state_shape),
    layers.Dense(24, activation='relu'),
    layers.Dense(num_actions, activation='linear')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

# Experience Replay
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.size = 0

    def add(self, experience):
        if self.size >= self.max_size:
            self.buffer.pop(0)
        else:
            self.size += 1
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.size), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return map(np.array, zip(*batch))

replay_buffer = ReplayBuffer(max_size=2000)

def train_step(batch_size):
    if replay_buffer.size < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    target_qs = model.predict(next_states)
    targets = rewards + gamma * np.max(target_qs, axis=1) * (1 - dones)
    target_f = model.predict(states)
    target_f[np.arange(batch_size), actions] = targets

    model.train_on_batch(states, target_f)

epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_shape[0]])
    total_reward = 0

    for step in range(200):
        if np.random.rand() <= epsilon:
            action = np.random.choice(num_actions)
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_shape[0]])
        replay_buffer.add((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
            break

        train_step(batch_size)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()
