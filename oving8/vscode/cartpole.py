import gymnasium as gym
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class QLearnCartPoleSolver():
    def __init__(self, env, buckets=(6, 12), episodes=100, epsilon_decay_rate=0.1,
                 decay=24, max_steps=100, batch_size=64, min_lr=0.1, discount=1.0, min_epsilon=0.1):

        self.env = env
        self.action_size = self.env.action_space.n
        self.discount = discount
        self.buckets = buckets
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.episodes = episodes
        self.decay = decay
        self.epsilon_decay_rate = epsilon_decay_rate
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.Q_Values = np.zeros(self.buckets + (self.action_size,))
        self.upper_bounds = [
            self.env.observation_space.high[2], math.radians(50)]
        self.lower_bounds = [
            self.env.observation_space.low[2], -math.radians(50)]

    # Epsilon (exploration rate) er verdien for sannsynligheten for en tilfeldig handling. Vår agent bruker det for å bestemme sitt neste trekk. For eks. om den skal
    # utforske miljøet eller fokusere på belønningen. Vi bruker Epsilon greedy strategy for å balansere mellom utfoskning og belønning.
    # Vi setter epsilon lik 1 og sakte senker den med decay verdien ettersom agenten har utforsket mer og mer.
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    # Hvor mye vekt vi legger på gamle q-verdiene. alpha = 1 betyr at vi kalkulerer de nye q-verdiene og ikke tar de gamle verdiene i betraktning i det hele tatt
    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def action(self, state):
        exploration_rate_threshold = np.random.random() # om agenten skal utforske eller fokusere på belønning
        return self.env.action_space.sample() if exploration_rate_threshold <= self.epsilon else np.argmax(self.Q_Values[state])

    # Oppdateringsregel for nye q-verdiene basert på Bellman-likning.
    # Q-verdiene må oppdateres for å gjenspeile endringen og forventningen til agenten som nå har utforsket mer av miljøet.
    # Nye q-verdien består av:
    # -learning_rate: læringsrate
    # -reward: belønning
    # -discount: hvor mye verdsetter vi fremtidig belønning (1 = bryr oss ikke om tid, 0= ingen belønnning, 0.1-0.9 er bra)
    # -max(Q_Values[state] - Q_Values[state][action]) fremtidig belønning, hva er neste oppnåelige belønningen.
    def updated_q_value(self, state, action, reward, new_state):
        return (self.learning_rate * (reward + self.discount * np.max(self.Q_Values[new_state]) - self.Q_Values[state][action]))

    def discretize_state(self, state):
        est = KBinsDiscretizer(n_bins=self.buckets,
                               encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds, self.upper_bounds])
        return tuple(map(int, est.transform([state[2:]])[0]))

    # Q-learning algorithm
    def train(self):
        rewards = []

        for episode in range(self.episodes):
            self.learning_rate = self.get_learning_rate(episode)
            self.epsilon = self.get_epsilon(episode)
            observation, info = self.env.reset()
            state = self.discretize_state(observation)
            done = False
            reward_current_ep = 0 # reward within current episode

            while not done: # do steps until episode is done
                action = self.action(state)
                new_state, reward, done, truncated, info = env.step(action)
                new_state = self.discretize_state(new_state)
                self.Q_Values[state][action] += self.updated_q_value(
                    state, action, reward, new_state)
                state = new_state
                reward_current_ep += 1
            rewards.append(reward_current_ep)
            print(f"{rewards[episode]}  score for ep {episode+1}")

        print('Finished training!')
        return rewards

    def run(self):
        done = False
        observation, info = self.env.reset()
        current_state = self.discretize_state(observation)
        score = 0
        while not done:
            self.env.render()
            action = self.action(current_state)
            observation, reward, done, truncated, info = self.env.step(action)
            new_state = self.discretize_state(observation)
            current_state = new_state
            score += reward
        print(f"score {score}")
        self.env.close()


print("MODEL TRAIN")
env = gym.make('CartPole-v1', render_mode="human")
model = QLearnCartPoleSolver(env, episodes=100)
rewards = model.train()

print("MODEL RUN")
model.run()
