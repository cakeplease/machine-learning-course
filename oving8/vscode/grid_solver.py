import math
import numpy as np

START=(2,0)
BOARD_ROWS=3
BOARD_COLS=4
DETERMINISTIC=True
WIN_STATE=(0,3)

class State():
    def __init__(self, state=START):
            self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
            self.state = state
            self.done = False
            self.determine = DETERMINISTIC
           

    def giveReward(self):
        if self.state == WIN_STATE:
            return 1
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.done = True

    def nextPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        if self.determine:
            if action == "up":
                nextState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nextState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nextState = (self.state[0], self.state[1] - 1)
            else:
                nextState = (self.state[0], self.state[1] + 1)
            # if next state legal
            if (nextState[0] >= 0) and (nextState[0] <= (BOARD_ROWS - 1)):
                if (nextState[1] >= 0) and (nextState[1] <= (BOARD_COLS - 1)):
                    if nextState != (1, 1):
                        return nextState
            return self.state


class QLearnGridSolver():
    def __init__(self, buckets=(6, 12), episodes=100, decay=24, batch_size=64, min_lr=0.1, discount=1.0, min_epsilon=0.1):

        self.states = []
        self.State = State()
        self.actions = ["up", "down", "left", "right"]
        self.Q_Values = []
        self.state_values = {}
        self.discount = discount
        self.buckets = buckets
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.episodes = episodes
        self.decay = decay
        self.batch_size = batch_size
        self.lr = 0.2
        self.exp_rate = 0.3
        

    def reset(self):
        self.states = []
        self.State = State()

    def get_epsilon(self, episode):
        return max(self.min_epsilon, min(1., 1. - math.log10((episode + 1) / self.decay)))

    def get_learning_rate(self, episode):
        return max(self.min_lr, min(1., 1. - math.log10((episode + 1) / self.decay)))

    def chooseAction(self):
        exploration_rate_threshold = np.random.random()  # tilfeldig float fra 0-1
        exploration_rate = self.epsilon
        mx_nxt_reward = 0
        action = ""
        if (exploration_rate_threshold <= exploration_rate):
            action = np.random.choice(self.actions)
        else:
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        
        return action
    
    def updated_q_value(self, state, action, reward, new_state):
        return (self.learning_rate * (reward + self.discount * np.max(self.Q_Values[new_state]) - self.Q_Values[state][action]))

    def takeAction(self, action):
        position = self.State.nextPosition(action)
        return State(state=position)
    
    # Q-learning algorithm
    def train(self):
        #episodes
        for episode in range(100):
            self.learning_rate = self.get_learning_rate(episode)
            self.epsilon = self.get_epsilon(episode)
            
            reward_current_ep = 0
            state = 1
            #steps
            while not self.State.done:
                action = self.chooseAction()
                self.states.append(self.State.nextPosition(action))
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                self.State.isEndFunc()
                print("next state", self.State.state)
                print("---------------------")

                #self.State = self.step(action)
                #self.Q_Values[state][action] += self.updated_q_value(state, action, self.State, new_state)
                #state = new_state
                #reward_current_ep += 1

            reward = self.State.giveReward()
            self.state_values[self.State.state] = reward
            print("Game End Reward", reward)
            for s in reversed(self.states):
                print("what")
                print(s)
                reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                
                self.state_values[s] = round(reward, 3)
            self.reset()
            # rewards.append(reward_current_ep)
            # print(f"Score for episode {episode+1}: {rewards[episode]}")

        print('Finished')
        #return rewards

    def play(self, rounds=10):
        i = 0
        
        while i < rounds:
            self.learning_rate = self.get_learning_rate(i)
            self.epsilon = self.get_epsilon(i)
            # to the end of game back propagate reward
            if self.State.done:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                # this is optional
                self.state_values[self.State.state] = reward
                print("Game End Reward", reward)
                print(self.state_values)
                for s in reversed(self.states):
                    reward = self.state_values[s] + \
                        self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nextPosition(action))
                print("current position {} action {}".format(
                    self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                print("nxt state", self.State.state)
                print("---------------------")

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


gridSolver = QLearnGridSolver()
gridSolver.play()
    # def run(self):
    #     done = False
    #     observation, info = self.env.reset()
    #     current_state = self.discretize_state(observation)
    #     score = 0
    #     while not done:
    #         #self.env.render()
    #         action = self.action(current_state)
    #         observation, reward, done = self.env.step(action)
    #         new_state = self.discretize_state(observation)
    #         current_state = new_state
    #         score += reward
    #     print(f"score {score}")
    #     self.env.close()

