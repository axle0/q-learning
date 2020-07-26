import random
import time

import numpy as np

from gridworld_functions import draw_qs, render_agent, create_environment, draw_policy


class QLearner:

    def __init__(self):
        # called to initialise the agent
        # set the Q_table to this objects variable
        self.q_table = np.zeros((5, 5, 4))  # 5 by 5 states + each state 4 action possibilities

        # initialize image holder
        self.q_image = None

        # calls the initialise function
        self.init_vars()

    def init_vars(self):
        self.epsilon = 1            # epsilon = 0 --> Exploitation | epsilon = 1 --> Exploration
        self.gamma = 0.95           # gamma = 0 --> Complete discounting! | gamma = 1 --> No discounting
        self.alpha = 0.5            # running average value (learning rate)
        self.decay_rate = 1.00005   # Used to decay the epsilon value.

    # The Q value at specified state and action (from the q table)
    def getQvalue(self, state, action):
        return self.q_table[state[0]][state[1]][action]

    # Find the max q-value in the passed state (which is the next-state relative to out current state)
    def getValue(self, nextstate):
        # find action leading to max q-value via the getPolicy() function
        next_action = self.getPolicy([nextstate[0], nextstate[1]])
        # find the q-value at the found action
        next_action_Qvalue = self.getQvalue([nextstate[0], nextstate[1]], next_action)
        return next_action_Qvalue

    # evaluate the action to take based on the epsilon value (epsilon greedy approach)
    def getAction(self, state):
        direction = ["left", "up", "right", "down"]
        # Generate a random value between 0.0 and 1.0 to compare with epsilon in order to find an action
        x = random.random()

        # if random number > self.epsilon : act greedily
        # if random number <= self.epsilon : act randomly
        if x > self.epsilon:
            next_action = int(self.getPolicy(state))
            return next_action
        else:
            next_action = random.randint(0, 3)
            return next_action

    # find best action to take in this state ( max_a(Q(s, a)) )
    def getPolicy(self, state):
        # We want the action yielding the highest result. But instead of finding that value ( amax() ),
        # we need its index ( argmax() ) or the action leading to max q-value
        return np.argmax(self.q_table[state[0]][state[1]])

    # update the q table using the bellman equation
    def update_Qvalues(self, state, action, reward, nextstate, done):
        if not done:
            q_value = (1 - self.alpha) * self.getQvalue(state, action) + self.alpha * (
                        reward + (self.gamma * self.getValue(nextstate)))
        else:
            # If at a terminal state, do not consider future values
            q_value = (1 - self.alpha) * self.getQvalue(state, action) + self.alpha * reward

        self.q_table[state[0]][state[1]][action] = q_value

    def call_draw_function(self, i):
        # this function passes the q_table to the draw_qs() function in the gridworld_functions.py file
        # it returns an image which is stored in self.q_image
        # then calls printQs() to display this image
        self.q_image = draw_qs(np.around(self.q_table, 4))
        self.printQs()
        string_test = "q_image_iteration_" + str(i) + ".png"
        self.q_image.save(string_test)

    def printOptimalPolicy(self):
        # you will need to print the optimal policy in the form step i , action : optimal_action where i is the ith
        # step in the environment and optimal_action is the action with the highest Q value for that state you can
        # access the variables env, mines, goal directly from this function, there is no need to pass them as inputs
        # to this function

        full_direction = ["left", "up", "right", "down"]
        direction = ["l", "u", "r", "d"]

        policy = np.empty((5, 5), dtype=str)

        # finding best policy for all states
        for n in range(5):
            for m in range(5):
                if env[n][m] == 'A' or env[n][m] == '-':
                    policy[n][m] = direction[int(self.getPolicy([n, m]))]
                else:
                    policy[n][m] = 'x'

        # finding best policy (best path)
        i = 0
        x, y = 0, 0
        while not (x == 4 and y == 4):
            policy[x][y] = policy[x][y].capitalize()
            print("Step ", i + 1, " , action:", full_direction[int(self.getPolicy([x, y]))])
            if policy[x][y] == 'U':
                x = x - 1
            elif policy[x][y] == 'D':
                x = x + 1
            elif policy[x][y] == 'L':
                y = y - 1
            elif policy[x][y] == 'R':
                y = y + 1
            i = i + 1
            # While should definitely not exceed 100 iterations. If so, no optimal policy was found and more episodes
            # are required
            if i > 100:
                print("Seems there's no valid optimal policy! Ensure enough episodes are set")
                exit()

        policy = np.array(policy)
        self.q_image = draw_policy(policy)
        self.printQs()
        self.q_image.save("optimal_policy.png")

    def printQs(self):
        self.q_image.show()


    def decay_exploration_prob(self):
        # Decay epsilon as specified by the decay_rate value until in reaches 0.1.
        # Decaying model used is exponential as opposed to linear
        if self.epsilon > 0.1:
            self.epsilon = self.epsilon / self.decay_rate


# main function
if __name__ == "__main__":
    # seed 'random' via time
    random.seed(time.time())

    # initialize the agent
    agent = QLearner()

    # draw the Qs to check initial values
    agent.call_draw_function(0)

    # initialize environment env stores a 2d array showing the environment when initialized and environment with
    # agents position when render_agent is called mines is a list that stores the positions of each mine as tuples
    # [(mine1xpos, mine1ypos), ..... (minelastxpos, minelastypos)]
    env, mines, goal = create_environment()

    # define the maximum episodes you want to run the simulation for
    max_episodes = 60000

    for ep in range(max_episodes):
        # reset agent's position (the agents State) and doneFlag
        current_state = [0, 0]
        isDone = False

        while not isDone:
            # pick an action for the agent
            action = agent.getAction(current_state)

            # take that action in the environment
            next_state, reward, isDone = render_agent(current_state, action, env, mines, goal)

            # now we have the (state, action, reward, nextstate, done) information
            # update the Q values in q_table with this information using the bellman equation
            agent.update_Qvalues(current_state, action, reward, next_state, isDone)

            # set the current state to next state for next iteration
            current_state = next_state

            # initially, if the getAction() function has not been filled, -1 will be returned
            # -1 is an invalid action, end episode
            if action == -1:
                isDone = True

        # decay the agent exploration probability
        agent.decay_exploration_prob()

    agent.call_draw_function(ep)
    agent.printOptimalPolicy()
