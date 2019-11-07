from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import math
import matplotlib.pyplot as plt
import numpy as np


class FlappyAgent:
    q = {}                  #lookup table for Q values
    learning_rate = 0.1     #alpha
    discount_factor = 0.9   #gamma
    epsilon = 0.1           #random factor
    episode = []            #episode used for monte carlo
    buckets = 15.0          #Amount of buckets to split every attribute in the state space
    initialQ = 0            #initial Q value changing this will make the agent more optimistic or realistic
    lastAction = 1          #Last action the agent did
    
    #dictionary for max values for each attributes, used when discretizing.
    maxs = {'player_y':512.0,
            'player_vel':19.0,
            'next_pipe_dist_to_player':150.0,
            'next_pipe_top_y':150.0,
            'next_pipe_bottom_y':412.0,
            'next_next_pipe_dist_to_player':288.0,
            'next_next_pipe_top_y':512.0,
            'next_next_pipe_bottom_y':412.0
            }                
    def __init__(self, learning_rate = 0.1, discount_factor = 0.9, epsilon = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        return
    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end): raise NotImplementedError("Override me")
    
    def training_policy(self, state):
        #epsilon amount chance to do a random action,  10% chance of flapping as a random action
        if random.random() >= self.epsilon:
            self.lastAction = self.policy(state)
        elif random.randint(1,10) == 10:
            self.lastAction = 0
        else:
            self.lastAction = 1
        return self.lastAction
    
    def policy(self, state):
        action = 0
        noFlap = 0.0
        flap = 0.0
        try:
            noFlap = self.q[(state, 1)]
        except KeyError:
            noFlap = 0.0
        try:
            flap = self.q[(state, 0)]
        except KeyError:
            flap = 0.0
        if flap > noFlap:
            action = 0
        # If values are equal 90% of noflap 10% chance of flap, sicne flap has way more significance
        elif flap == noFlap and random.randint(1,10) == 10:
            action = 0
        else:
            action = 1
        return action        
    def discretize_state(self, state): raise NotImplementedError("Override me")

class FlappyAgentMC(FlappyAgent):
    filename = 'MC.png'
    
    def observe(self, s1, a, r, s2, end):
        #If it's not the end of the episode append the state action pair and reward
        self.episode.append([s1,a,r])
        if end:
            G = 0
            for s,a,r in self.episode[::-1]:
                #Monte carlo value function
                G = r + self.discount_factor * G
                try:
                    self.q[(s,a)] = self.q[(s,a)] + self.learning_rate*(G - self.q[(s,a)])
                except KeyError:
                    self.q[(s,a)] = G
            self.episode = []
        return

    def discretize_state(self, state):
        dstate = (math.floor(state['player_y']*self.buckets/self.maxs['player_y']),
                math.floor(state['player_vel']),
                math.floor(state['next_pipe_dist_to_player']*self.buckets/self.maxs['next_pipe_dist_to_player']),
                math.floor(state['next_pipe_top_y']*self.buckets/self.maxs['next_pipe_top_y']))
        return dstate

class FlappyAgentQ(FlappyAgent):
    filename = 'Q.png'
    def observe(self, s1, a, r, s2, end):
        try:
            #Q-learning value function
            if not end:
                aprime = self.policy(s2)
                self.q[(s1,a)] = self.q[(s1,a)] + self.learning_rate * (r + self.discount_factor * self.q[(s2,aprime)] - self.q[(s1,a)])
            else:
                self.q[(s1,a)] = self.q[(s1,a)] + self.learning_rate * (r - self.q[(s1,a)])
        except KeyError:
                    self.q[(s1,a)] = self.initialQ
        return

    def discretize_state(self, state):
        dstate = (math.floor(state['player_y']*self.buckets/self.maxs['player_y']),
                math.floor(state['player_vel']),
                math.floor(state['next_pipe_dist_to_player']*self.buckets/self.maxs['next_pipe_dist_to_player']),
                math.floor(state['next_pipe_top_y']*self.buckets/self.maxs['next_pipe_top_y']))
        return dstate

class FlappyAgentBest(FlappyAgent):
    filename = 'Best.png'
    def observe(self, s1, a, r, s2, end):
        try:
            if not end:
                aprime = self.policy(s2)
                self.q[(s1,a)] = self.q[(s1,a)] + self.learning_rate * (r + self.discount_factor * self.q[(s2,aprime)] - self.q[(s1,a)])
            else:
                self.q[(s1,a)] = self.q[(s1,a)] + self.learning_rate * (r - self.q[(s1,a)])
        except KeyError:
                    self.q[(s1,a)] = self.initialQ
        return

    def discretize_state(self, state):
        stable = 30
        pipeDeltaY = 0
        if state['next_pipe_top_y'] - state['next_next_pipe_top_y'] -50 < 0:
            pipeDeltaY = -1
        if state['next_pipe_top_y'] - state['next_next_pipe_top_y'] -50 > 0:
            pipeDeltaY = 1
        deltaY = state['player_y'] - state['next_pipe_top_y'] - stable
        dist = state['next_pipe_dist_to_player']
        playerVel = 1
        if state['player_vel'] < 0:
            playerVel = -1
        if dist < 10:
            deltaY = state['player_y'] - state['next_next_pipe_top_y'] - stable
        if deltaY > 1:
            deltaY = 1
        elif deltaY < -1:
            deltaY = -1
        else:
            deltaY = 0
        dstate = (deltaY, playerVel, pipeDeltaY, self.lastAction)
        return dstate

class FlappyAgentLFA(FlappyAgent):
    filename = 'LFA.png'
    flapv = [0,0,0,0]
    noflapv = [0,0,0,0]
    
    def observe(self, s1, a, r, s2, end):
        self.episode.append([s1,a,r])
        if end:
            G = 0
            for s,a,r in self.episode[::-1]:
                G = r + self.discount_factor * G
                if a == 0:
                    for i in range(len(self.flapv)):
                        self.flapv[i] = self.flapv[i] + self.learning_rate * (G - self.flapv[i]) * s[i]
                else:
                    for i in range(len(self.noflapv)):
                        self.noflapv[i] = self.noflapv[i] + self.learning_rate * (G - self.noflapv[i]) * s[i]
            self.episode = []

    def policy(self, state):
        action = 1
        flap = np.dot(self.flapv, state)
        noflap = np.dot(self.noflapv, state)
        if flap > noflap:
            action = 0
        return action

    def discretize_state(self, state):
        return (state['player_y'], state['player_vel'], state['next_pipe_top_y'], state['next_pipe_dist_to_player'])

def run_game(nb_episodes, agent):
    reward_values = agent.reward_values()
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None, reward_values = reward_values)
    env.init()
    maxScore = 0    #Highscore
    score = 0       #Current score
    test = 0        #Amount test left
    frames = 0      #Frame counter
    acScore = 0     #Score accumulated
    testAcScore = 0     #Score accumulated for testing
    trainingEpisodes = 100  #Amount of episode to train before testing
    testingEpisodes = 10    #Amount of testing episodes in each test
    avgScore = 0            #Average score
    avgScoresArray = []     #Average score list for the plot
    framesArray = []        #Frames for the plot
    while nb_episodes > 0:
        action = 0
        # start by discretizing and calling the policy
        state = agent.discretize_state(env.game.getGameState())
        if test > 0:
            action = agent.policy(state)
        else:
            action = agent.training_policy(state)
        #Now we have a state action pair, we use the action to act on the environment
        reward = env.act(env.getActionSet()[action])
        
        #plotting
        if frames % 1000 == 0 and frames != 0:
            avgScore = acScore/(runs - nb_episodes + 1)
            avgScoresArray.append(avgScore)
            framesArray.append(frames)
            plt.plot(framesArray, avgScoresArray)
            plt.savefig(agent.filename)
        
        frames += 1
        if reward > 0:
            score += reward
            acScore += reward
            testAcScore += reward

        #This bird got far, lets watch it
        if score == 2000:
            env.display_screen = True
            env.force_fps = False
        #Bird is pretty good update us on every 1000 score just to rougly know how he's doing
        if score % 1000 == 0 and score != 0:
            print('episode:',(runs - nb_episodes),'Big score', score)
        statePrime = agent.discretize_state(env.game.getGameState())

        #dont update while testing
        if test <= 0:
            agent.observe(state, action, reward, statePrime, env.game_over())

        if env.game_over():
            if (runs - nb_episodes) % trainingEpisodes == (trainingEpisodes - 1):
                #uncomment to see how he is doing while testing
                # env.display_screen = True
                # env.force_fps = False
                test = testingEpisodes
                print('State space:', len(agent.q))
                testAcScore = 0

                #decrease learning rate over time
                agent.learning_rate /= 2
            elif test > 0:
                test -= 1
            else:
                env.display_screen = False
                env.force_fps = True

            #New highscore
            if score > maxScore:
                maxScore = score
                print("Highscore:", maxScore)
            if test > 0:
                avgScore = testAcScore/((testingEpisodes+1) - test)
                print("Highscore:", maxScore, "Average:", format(avgScore, '.3f'), "Frame:", frames, "Episode:", runs - nb_episodes + 1, " Score:", score)
            if frames == 1000000:
                print("*****************************************************************************\nFrame limit reached\n**********************************************************")
            env.reset_game()
            nb_episodes -= 1
            score = 0




runs = 100000
alpha = 0.1
gamma = 0.1
epsilon = 0.1

# agent = FlappyAgentMC(alpha,gamma,epsilon)
# agent = FlappyAgentQ(alpha,gamma,epsilon)
# agent = FlappyAgentLFA(alpha,gamma,epsilon)
agent = FlappyAgentBest(0.1, 0.9, 0.0)
run_game(runs, agent)
