from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import math
class FlappyAgent:
    q = {}
    learning_rate = 0.2
    discount_factor = 0.999
    epsilon = 0.1
    episode = []
    buckets = 15.0
    newKey = 0
    initialQ = 0
    lastAction = 1
    maxs = {'player_y':512.0,
            'player_vel':18.0,
            'next_pipe_dist_to_player':288.0,
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
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        # print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        action = 0
        if random.random() >= self.epsilon:
            noFlap = 0.0
            flap = 0.0
            try:
                noFlap = self.q[(state, 1)]
            except KeyError:
                # print((state,1), end='')                
                # self.q[state, 1] = 0
                noFlap = -1000
            try:
                flap = self.q[(state, 0)]
            except KeyError:
                # self.q[state, 0] = 0
                # print((state,0), end='')                
                flap = -1000
            if flap > noFlap:
                action = 0
            # elif flap == noFlap:
            #     action = random.randint(0, 1) 
            else:
                action = 1
        else:
            action = random.randint(0, 1) 
        return action
        

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        return self.training_policy(state)
        # print("state: %s" % state)
        # # TODO: change this to to policy the agent has learned
        # # At the moment we just return an action uniformly at random.
        # return random.randint(0, 1) 
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
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
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
            return self.policy(state)
        elif random.randint(1,10) == 10:
            return 0
        return 1
        

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
        self.lastAction = action
        return action

    def discretize_state(self, state):
        #state: player_y, player_vel, next_pipe_dist_to_player, next_pipe_top_y, next_pipe_bottom_y, next_next_pipe_dist_to_player, next_next_pipe_top_y, next_next_pipe_bottom_y
        deltaY = state['player_y'] - state['next_pipe_top_y'] - 30
        dist = state['next_pipe_dist_to_player']
        playerVel = 1
        pipeDeltaY = 1
        if state['player_vel'] < 0:
            playerVel = -1
        if state['next_pipe_top_y'] - state['next_next_pipe_top_y'] < 0:
            pipeDeltaY = -1
        if dist < 15:
            deltaY = state['player_y'] - state['next_next_pipe_top_y'] - 30
        if deltaY > 1:
            deltaY = 1
        elif deltaY < -1:
            deltaY = -1
        dstate = (deltaY
                #math.floor(deltaY*self.buckets/self.maxs['next_pipe_top_y'])
                ,playerVel
                # ,math.floor(dist*self.buckets/self.maxs['next_pipe_dist_to_player'])
                ,pipeDeltaY,
                # ,math.floor(pipeDeltaY*self.buckets/self.maxs['next_pipe_top_y'])
                self.lastAction
                )
        # for i in range(len(dstate)):
        #     if dstate[i] == 0:
        #         print(i, end=' ')
        return dstate

def run_game(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, rng=None, reward_values = reward_values)
    env.init()
    maxScore = 0
    score = 0
    test = False
    frames = 0
    acScore = 0
    while nb_episodes > 0:
        action = 0
        state = agent.discretize_state(env.game.getGameState())
        if test:
            action = agent.policy(state)
        else:
            action = agent.training_policy(state)

        reward = env.act(env.getActionSet()[action])
        frames += 1
        if reward > 0:
            score += reward
        statePrime = agent.discretize_state(env.game.getGameState())
        agent.observe(state, action, reward, statePrime, env.game_over())
            
        if env.game_over():
            if (runs - nb_episodes) % 100 == 99:
                env.display_screen = True
                env.force_fps = False
                test = not test
                acScore = 0
                print(len(agent.q))

                agent.learning_rate /= 2
            else:
                env.display_screen = False
                env.force_fps = True
            if score > maxScore:
                maxScore = score
            if score > 0:
                acScore += score
                avgScore = acScore/((runs - nb_episodes ) % 100 + 1)
                print("Highscore:", maxScore, "Average:", format(avgScore, '.3f'), "Keys:", agent.newKey, "Frame:", frames, "Episode:", runs - nb_episodes, " Score:", score)
            agent.newKey = 0
            if frames == 1000000:
                print("Frame limit reached")
            env.reset_game()
            nb_episodes -= 1
            score = 0




runs = 100000
# agent = FlappyAgentMC()
# agent = FlappyAgentQ()
agent = FlappyAgentBest()
run_game(runs, agent)

