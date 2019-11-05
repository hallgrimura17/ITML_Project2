from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import math
class FlappyAgent:
    q = {}
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.01
    episode = []
    buckets = 15.0
    newKey = 0
    initialQ = 0
    lastAction = 1
    maxs = {'player_y':512.0,
            'player_vel':18.0,
            'next_pipe_dist_to_player':150.0,
            'next_pipe_top_y':150.0,
            'next_pipe_bottom_y':412.0,
            'next_next_pipe_dist_to_player':288.0,
            'next_next_pipe_top_y':512.0,
            'next_next_pipe_bottom_y':412.0
            }
    def __init__(self):
        return
    def reward_values(self): raise NotImplementedError("Override me")
    def observe(self, s1, a, r, s2, end): raise NotImplementedError("Override me")
    def training_policy(self, state): raise NotImplementedError("Override me")
    def policy(self, state): raise NotImplementedError("Override me")
    def discretize_state(self, state): raise NotImplementedError("Override me")

class FlappyAgentMC(FlappyAgent):
    def __init__(self):
        return
    
    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        self.episode.append([s1,a,r])
        # print('h', end='')
        if end:
            G = 0
            for s,a,r in self.episode[::-1]:
                # print(s,a,r)
                G = r + self.discount_factor * G
                try:
                    self.q[(s,a)] = self.q[(s,a)] + self.learning_rate*(G - self.q[(s,a)])
                except:
                    self.q[(s,a)] = G
                    # print((s,a))
                # if self.q[(s,a)] != 0:
                #     print(self.q[(s,a)], end=' ')
            self.episode = []
        return

    def training_policy(self, state):
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
            self.q[(state, 0)] = self.initialQ
            noFlap = -1000.0
        try:
            flap = self.q[(state, 0)]
        except KeyError:
            self.q[(state, 0)] = self.initialQ
            flap = -1000.0
        if flap > noFlap:
            action = 0
        elif flap == noFlap and random.randint(1,10) == 10:
            action = 0
        else:
            action = 1
        return action

    def discretize_state(self, state):
        #state: player_y, player_vel, next_pipe_dist_to_player, next_pipe_top_y, next_pipe_bottom_y, next_next_pipe_dist_to_player, next_next_pipe_top_y, next_next_pipe_bottom_y
        dstate = (math.floor(state['player_y']*self.buckets/self.maxs['player_y']),
                math.floor(state['player_vel']),
                math.floor(state['next_pipe_dist_to_player']*self.buckets/self.maxs['next_pipe_dist_to_player']),
                math.floor(state['next_pipe_top_y']*self.buckets/self.maxs['next_pipe_top_y']))
        
        # for x in dstate:
            # if x >= 16:
                # print(x)
        return dstate

class FlappyAgentQ(FlappyAgent):
    def __init__(self):
        return
    
    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        if not end:
                try:
                    aprime = self.policy(s2)
                    self.q[(s1,a)] = self.q[(s1,a)] + self.learning_rate * (r + self.discount_factor * self.q[(s2,aprime)] - self.q[(s1,a)])
                except KeyError:
                    self.q[(s1,a)] = r
                    self.newKey += 1
        else:
                try:
                    self.q[(s1,a)] = self.q[(s1,a)] + self.learning_rate * (r - self.q[(s1,a)])
                except KeyError:
                    self.q[(s1,a)] = r
        return

    def training_policy(self, state):
        if random.random() >= self.epsilon:
            return self.policy(state)
        else:
            return random.randint(0, 1) 
        

    def policy(self, state):
        action = 0
        noFlap = 0.0
        flap = 0.0
        try:
            noFlap = self.q[(state, 1)]
        except KeyError:
            noFlap = -1000.0
        try:
            flap = self.q[(state, 0)]
        except KeyError:
            flap = -1000.0
        if flap > noFlap:
            action = 0
        elif flap == noFlap and random.randint(1,10) == 10:
            action = 0
        else:
            action = 1
        return action

    def discretize_state(self, state):
        #state: player_y, player_vel, next_pipe_dist_to_player, next_pipe_top_y, next_pipe_bottom_y, next_next_pipe_dist_to_player, next_next_pipe_top_y, next_next_pipe_bottom_y
        dstate = (math.floor(state['player_y']*self.buckets/self.maxs['player_y']),
                math.floor(state['player_vel']),
                math.floor(state['next_pipe_dist_to_player']*self.buckets/self.maxs['next_pipe_dist_to_player']),
                math.floor(state['next_pipe_top_y']*self.buckets/self.maxs['next_pipe_top_y']))
        return dstate

class FlappyAgentBest(FlappyAgent):
    def __init__(self):
        return
    
    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -50.0}
    
    def observe(self, s1, a, r, s2, end):
        try:
            if not end:
                aprime = self.policy(s2)
                self.q[(s1,a)] = self.q[(s1,a)] + self.learning_rate * (r + self.discount_factor * self.q[(s2,aprime)] - self.q[(s1,a)])
            else:
                self.q[(s1,a)] = self.q[(s1,a)] + self.learning_rate * (r - self.q[(s1,a)])
        except KeyError:
                    self.q[(s1,a)] = self.initialQ
                    self.newKey += 1
        return

    def training_policy(self, state):
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
            self.q[(state, 0)] = self.initialQ
            noFlap = -1000.0
        try:
            flap = self.q[(state, 0)]
        except KeyError:
            self.q[(state, 0)] = self.initialQ
            flap = -1000.0
        if flap > noFlap:
            action = 0
        elif flap == noFlap and random.randint(1,10) == 10:
            action = 0
        else:
            action = 1
        return action

    def discretize_state(self, state):
        #state: player_y, player_vel, next_pipe_dist_to_player, next_pipe_top_y, next_pipe_bottom_y, next_next_pipe_dist_to_player, next_next_pipe_top_y, next_next_pipe_bottom_y
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
        if dist < 15:
            deltaY = state['player_y'] - state['next_next_pipe_top_y'] - stable
        if deltaY > 1:
            deltaY = 1
        elif deltaY < -1:
            deltaY = -1
        dstate = (deltaY
                # math.floor(deltaY*self.buckets/self.maxs['next_pipe_top_y'])
                ,playerVel
                # ,math.floor(dist*self.buckets/self.maxs['next_pipe_dist_to_player'])
                ,pipeDeltaY
                # ,math.floor(pipeDeltaY*self.buckets/self.maxs['next_pipe_top_y'])
                ,self.lastAction
                )
        # for i in range(len(dstate)):
        #     if dstate[i] == 0:
        #         print(i, end=' ')
        return dstate

class FlappyAgentLFA(FlappyAgent):
    def __init__(self):
        return
    
    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        self.episode.append([s1,a,r])
        if end:
            G = 0
            for s,a,r in self.episode[::-1]:
                G = r + self.discount_factor * G
                try:
                    self.q[(s,a)] = self.q[(s,a)] + self.learning_rate*(G - self.q[(s,a)])
                except:
                    self.q[(s,a)] = G
            self.episode = []
        return

    def training_policy(self, state):
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
            self.q[(state, 0)] = self.initialQ
            noFlap = -1000.0
        try:
            flap = self.q[(state, 0)]
        except KeyError:
            self.q[(state, 0)] = self.initialQ
            flap = -1000.0
        if flap > noFlap:
            action = 0
        elif flap == noFlap and random.randint(1,10) == 10:
            action = 0
        else:
            action = 1
        return action

    def linear_regression(self, X, y, m_current=0, b_current=0, epochs=1000):
        N = float(len(y))
        for i in range(epochs):
            y_current = (m_current * X) + b_current
            m_gradient = -(2/N) * sum(X * (y - y_current))
            b_gradient = -(2/N) * sum(y - y_current)
            m_current = m_current - (self.learning_rate * m_gradient)
            b_current = b_current - (self.learning_rate * b_gradient)
        return m_current, b_current

    def discretize_state(self, state):
        return state['player_y'] - state['next_pipe_top_y']

def run_game(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None, reward_values = reward_values)
    env.init()
    maxScore = 0
    score = 0
    test = 0
    frames = 0
    acScore = 0
    trainingEpisodes = 100
    testingEpisodes = 10
    while nb_episodes > 0:
        action = 0
        state = agent.discretize_state(env.game.getGameState())
        if test > 0:
            action = agent.policy(state)
        else:
            action = agent.training_policy(state)

        reward = env.act(env.getActionSet()[action])
        frames += 1
        if reward > 0:
            score += reward
        # statePrime = env.game.getGameState()
        # if env.game_over():
        #     reward = -abs(statePrime['player_y'] - statePrime['next_pipe_top_y'] - 50)
        statePrime = agent.discretize_state(env.game.getGameState())
        agent.observe(state, action, reward, statePrime, env.game_over())
        # if action == 0:
        #     reward += env.act(env.getActionSet()[1])   
        if env.game_over():
            if (runs - nb_episodes) % trainingEpisodes == (trainingEpisodes - 1):
                env.display_screen = True
                env.force_fps = False
                test = testingEpisodes
                acScore = 0
                print('State space:',len(agent.q))

                agent.learning_rate /= 2
            elif test > 0:
                test -= 1
            else:
                # env.display_screen = False
                env.force_fps = True

            if score > maxScore:
                maxScore = score
                print("Highscore:", maxScore)
            if test > 0:
                acScore += score
                avgScore = acScore/((testingEpisodes+1) - test)
                print("Highscore:", maxScore, "Average:", format(avgScore, '.3f'), "Keys:", agent.newKey, "Frame:", frames, "Episode:", runs - nb_episodes + 1, " Score:", score)
            agent.newKey = 0
            if frames == 1000000:
                print("Frame limit reached")
            env.reset_game()
            nb_episodes -= 1
            score = 0




runs = 100000
# agent = FlappyAgentMC()
# agent = FlappyAgentQ()
# agent = FlappyAgentLFA()
agent = FlappyAgentBest()
run_game(runs, agent)

# a = 0.4
# e = 0.25
# g = 0.9
# q = {'a,up':0, 'b,up':0, 'c,up':0, 'd,right':0, 'e,right':0,'f,down':0}
# lastState = 'a,up'
# for i in range(10000):
#     for s,r in q.items():
#         if s == 'f,down':
#             q[lastState] = q[lastState] + a*(0 + g*q[s] - q[lastState])
#             q[s] = q[s] + a*(10 - q[s])
#         else:
#             q[lastState] = q[lastState] + a*(0 + g*q[s] - q[lastState])
#             lastState = s
#     for x in q.items():
#         if x[1] != 0:
#             print(x)
#     print()
