from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import math
class FlappyAgent:
    q = {}
    learning_rate = 0.1
    discount_factor = 0.999
    epsilon = 0.1
    episode = []
    buckets = 15.0
    maxs = {'player_y':513.0,
            'player_vel':19.0,
            'next_pipe_dist_to_player':288.0,
            'next_pipe_top_y':513.0,
            'next_pipe_bottom_y':413.0,
            'next_next_pipe_dist_to_player':288.0,
            'next_next_pipe_top_y':513.0,
            'next_next_pipe_bottom_y':413.0
            }
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
            elif flap == noFlap:
                action = random.randint(0, 1) 
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

runs = 100000

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    # reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": -5.0, "win": 0.0}
    # TODO: when training use the following instead:
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()
    maxScore = 0
    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        state = agent.discretize_state(env.game.getGameState())
        action = agent.policy(state)
        # step the environment
        reward = env.act(env.getActionSet()[action])
        if reward > 0:
            score += reward

        # TODO: for training let the agent observe the current state transition
        agent.observe(state, action, reward, env.game.getGameState(), env.game_over())
        
        # reset the environment if the game is over
        if env.game_over():
            if score > maxScore:
                maxScore = score
                print("Highscore: Episode:", runs - nb_episodes, " Score:", score)
            elif score > 0:
                print("Episode:", runs - nb_episodes, " Score:", score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

agent = FlappyAgent()
run_game(runs, agent)
