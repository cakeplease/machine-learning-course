import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

MODEL_FILE_NAME = "dqncartpole.h5"
env = gym.make('CartPole-v0')
tf.random.set_seed(200)

gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)
tf.test.is_built_with_cuda()
state_space = env.observation_space.shape[0]
env.observation_space.shape
action_space = env.action_space.n
action_space


class DQNQLearnCartPoleSolver():
    def __init__(self, env,  input_shape, action_shape, episodes, epsilon_decay_rate=0.999, min_epsilon=0.001):
        self.input_size = input_shape
        self.episodes = episodes
        self.env = env
        self.action_size = action_shape
        self.memory = deque([], maxlen=20000)
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = 0.1
        self.state_size = input_shape
        self.batch_size = 64
        self.gamma = 0.99
        self.train_start = 1000
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=input_shape,
                       activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(action_shape, activation="linear",
                       kernel_initializer='he_uniform'))

        self.model.compile(loss="mse", optimizer=RMSprop(
            learning_rate=0.00025), metrics=["accuracy"])

    def action(self, state):
        # print(f" rand nr {np.random.random()}  eps {self.epsilon}")
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def preprocess_state(self, state):
        return np.reshape(state, [1, self.state_size])

    def update_q_func(self, reward, next_state, done):
        if done:
            return reward
        else:
            return reward + self.gamma * np.max(next_state)

    def update_q_values(self, minibatch, target, target_next):
        for index, (_, action, reward, _, done) in enumerate(minibatch):
            target[index][action] = self.update_q_func(
                reward, target_next[index], done)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(
            len(self.memory), self.batch_size))
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        for index, (state, _, _, next_state, _) in enumerate(minibatch):
            states[index] = state
            next_states[index] = next_state
        target = self.model.predict(states)
        target_next = self.model.predict(next_states)
        self.update_q_values(minibatch, target, target_next)
        self.model.fit(np.array(states), np.array(target),
                       batch_size=self.batch_size, verbose=0)
        self.update_epsilon()

    def get_reward(self, done, step, reward):
        if not done or step == self.env._max_episode_steps-1:
            return reward
        else:
            return -100

    def train(self):
        scores = []
        for episode in range(self.episodes):
            done = False
            state = self.preprocess_state(self.env.reset())
            step = 0
            while not done:
                # self.env.render()
                action = self.action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                step += 1
                self.remember(state, action, reward, next_state, done)
                reward = self.get_reward(done, step, reward)
                state = next_state
            scores.append(step)
            print(
                f"{scores[episode]}  score for ep {episode+1} epsilon {self.epsilon}")
            if step == 200:
                print(f"Saving trained model as {MODEL_FILE_NAME}")
                self.model.save(MODEL_FILE_NAME)
            self.replay()
        # self.env.close()
        print('Finished training!')

    def test(self):
        self.model = load_model(MODEL_FILE_NAME)
        state = self.preprocess_state(self.env.reset())
        done = False
        score = 0
        while not done:
            self.env.render()
            action = np.argmax(self.model.predict(state))
            next_state, reward, done, _ = self.env.step(action)
            state = self.preprocess_state(next_state)
            score += 1
        print(f"{score}  score")
        self.env.close()
