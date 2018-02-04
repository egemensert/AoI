""" Inspired by https://keon.io/deep-q-learning/ """

import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam

from collections import deque
import random

import matplotlib
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim = self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.randn() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

def plot_durations(durations_t, means):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t, alpha=0.3)
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means.append(np.array(durations_t[-100:]).mean())
        plt.plot(means)

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('training')

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = DQNAgent(4, 2)

    episodes = 5000
    means = [0 for i in range(99)]
    durations = []
    for e in range(episodes):
        state = env.reset()
        state = state.reshape((1, 4))
        for time_t in range(500):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, 4)

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            env.render()

            if done:
                durations.append(time_t)
                plot_durations(durations, means)
                break

        agent.replay(32)
