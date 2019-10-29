# Snake Tutorial Python
 
import math
import random
import pygame
import tkinter as tk
import copy
from tkinter import messagebox
from random import shuffle
from pygame import QUIT, quit, K_ESCAPE

# amount of rows and columns
rows = 20
# starting x position
startx = 10
# starting y position
starty = 15
# width of the window
w = 500
# lookup table for the agent
qsa = {}
# value for discovering a new state
initialStateValue = 0.0

# this is a piece of the snake and the food on the field
class Cube(object):
    def __init__(self,start,color=(255,0,0)):
        self.pos = start
        self.color = color
    def move(self, dirnx, dirny):
        self.pos = (self.pos[0] + dirnx, self.pos[1] + dirny)
    def draw(self, surface, eyes=False):
        dis = w // rows
        i = self.pos[0]
        j = self.pos[1]
 
        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))
        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre-radius,j*dis+8)
            circleMiddle2 = (i*dis + dis -radius*2, j*dis+8)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)
# Function to abstract the playing field that act as keys in the dictionary that is the lookup table for the policy  
def abstractState(snake):
    # minus one means that none of the item was found
    sens = [-1]*8
    x_pos = snake.body[0].pos[0]
    y_pos = snake.body[0].pos[1]

    # WALLS
    # this calculates the distance to all walls given the position of the snake
    sens[0] = x_pos + 1             # <
    sens[1] = rows - x_pos          # >
    sens[2] = y_pos + 1             # ^
    sens[3] = rows - y_pos          # v
    
    # this calculates the distance to a body part if there is one horizontally and vertically 
    for i in range(rows):
        if i != x_pos and (i, y_pos) in snake.locations:     # BODY X
            if i <= x_pos:
                sens[4] = x_pos - i
            if i >= x_pos:
                sens[5] = i - x_pos

        if i != y_pos and (x_pos, i) in snake.locations:     # BODY Y
            if i <= y_pos: 
                sens[6] = y_pos - i
            if i >= y_pos:
                sens[7] = i - y_pos
    # FOOD
    # Calculates the delta x and y from the snake to the food.
    # This is used so the snake has a sense of where the food is, so it doesnt randomly go in circles without achieving anything
    deltaX = snake.snack.pos[0] - x_pos
    deltaY = snake.snack.pos[1] - y_pos

    # limit distance of food so not too many different values exist for it, this abstracts the state and reduces state space.
    if deltaX < -2:
        deltaX = -2
    if 2 < deltaX:
        deltaX = 2
    if deltaY < -2:
        deltaY = -2
    if 2 < deltaY:
        deltaY = 2

    for i in range(8):  # 9 variables with distances approximated to near, medium and far; 3^5 * 4^5 = 248,832 
        # Distance class:     none seen         nearBy   <    mediumDistanse    <   farAway
        sens[i] = (sens[i] if sens[i] < 2 else (2 if sens[i] < 3 else (3 if sens[i] < 6 else 4)))
    # Here we rotate the sensors for horizontal and vertical to be relative to the snake and adjust the delta x and y to also relative
    # essentially we are rotating the field with snake as the center.
    if snake.dirnx == 1: 
               # (wall                    )# (body                   ) # (food          )
                # left    forward  right     left     forward  right    
        return (sens[2], sens[1], sens[3], sens[6], sens[5], sens[7], deltaY, -deltaX) 
    if snake.dirnx == -1:
        return (sens[3], sens[0], sens[2], sens[7], sens[4], sens[6], -deltaY, deltaX)
    if snake.dirny == 1:
        return (sens[1], sens[3], sens[0], sens[5], sens[7], sens[4], -deltaX, -deltaY)
    if snake.dirny == -1:
        return (sens[0], sens[2], sens[1], sens[4], sens[6], sens[5], deltaX, deltaY)

# our state object called it snake since its fitting and sounds like state
class Snake(object):
    body = []               # the visual body of the snake
    turns = {}              # What turns to execute, this is used with human controls
    locations = set()       # set that determines if the snake is colliding
    dirnx = 0               # x speed of the snake
    dirny = -1              # y speed of the snake
    oldtail = 0             # variable to add the old tail if it eats food
    lastAction = "f"        # action that previously changed the state
    isTraining = True       # whether to update the look up table and play the game slow
    epsilon = 0.1           # exploration factor
    snack = Cube((0,0))     # initialize the food
    ate = False             # tells the main function the snake ate food
    dead = False            # tells the main function its dead
    def randomSnack(self, rows):
        positions = self.body
        # food cant be in snake
        while True:
            x = random.randrange(rows)
            y = random.randrange(rows)
            if len(list(filter(lambda z:z.pos == (x,y), positions))) > 0:
                continue
            else:
                break
        return (x,y)

    def __init__(self, color, pos):
        self.color = color
        self.head = Cube(pos)
        self.body.append(self.head)
        self.locations.add(self.body[0].pos)
        #starting length of snake is 5
        for i in range(1, 5):
            self.body.append(Cube((pos[0],pos[1] + i)))
            self.locations.add(self.body[i].pos)
        self.dirnx = 0
        self.dirny = -1
        self.snack = Cube(self.randomSnack(rows), color=(0,255,0))
    
    def keyInput(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                quit()
 
        keys = pygame.key.get_pressed()

        #hold escape to quit
        if keys[K_ESCAPE]:
            quit()

    def move(self):
        #movement related functionality. moving the tail of sneak in front of it to make it seem like its moving
        if self.body[-1].pos in self.locations:
            self.locations.remove(self.body[-1].pos)
        self.oldtail = self.body[-1].pos
        self.body[-1].pos = (self.body[0].pos[0] + self.dirnx, self.body[0].pos[1] + self.dirny)
        self.head = self.body[-1]
        self.body = [self.head] + self.body[:-1]
        self.locations.add(self.head.pos)
        # the reward function 2*length of snake makes the agent more brave the longer it is. -10 for dying as soft punishment.
        reward = (2*len(self.body)*(float)(self.body[0].pos[0] == self.snack.pos[0] and self.body[0].pos[1] == self.snack.pos[1]) - 10*float(isGameOver(self)))
        # check whether the snake ate the food
        if self.body[0].pos == self.snack.pos:
            self.addCube()
            self.snack = Cube(self.randomSnack(rows), color=(0,255,0))
            self.ate = True
        # once the dead the main function will restart the game
        if isGameOver(self):
            self.dead = True
        return reward

    def reset(self, pos):
        self.locations = set()
        self.body = []
        self.turns = {}
        self.__init__((255,0,0), pos)

    #extent the snake once he eats
    def addCube(self):
        self.body.append(Cube(self.oldtail)) 
        self.locations.add(self.oldtail)
    
    # gets all moves in the state and their values
    def getMoves(self):
        moves = []
        #try so if the state has never been discovered before, the values is 0
        try:
            moves.append((qsa[(abstractState(self), "l")], "l"))
        except:
            moves.append((initialStateValue, "l"))
        try:
            moves.append((qsa[(abstractState(self), "f")], "f"))
        except:
            moves.append((initialStateValue, "f"))
        try:
            moves.append((qsa[(abstractState(self), "r")], "r"))
        except:
            moves.append((initialStateValue, "r"))
        shuffle(moves)
        return moves

    #change the direction of head so the move action moves the tail correctly
    def AdjustCourse(self, action):
        if action == "l":
            if self.dirnx == 1:
                self.dirny = -1
                self.dirnx = 0
            elif self.dirnx == -1:
                self.dirny = 1
                self.dirnx = 0
            elif self.dirny == 1:
                self.dirnx = 1
                self.dirny = 0
            elif self.dirny == -1:
                self.dirnx = -1
                self.dirny = 0
        elif action == "r":
            if self.dirnx == 1:
                self.dirny = 1
                self.dirnx = 0
            elif self.dirnx == -1:
                self.dirny = -1
                self.dirnx = 0
            elif self.dirny == 1:
                self.dirnx = -1
                self.dirny = 0
            elif self.dirny == -1:
                self.dirnx = 1
                self.dirny = 0

    # finds the best move of the state given the current policy
    def policy(self):
        self.AdjustCourse(self.lastAction)
        reward = self.move()
        moves = self.getMoves()
        move = moves[0]
        #epsilon amount of chance to pick a random move
        if random.random() >= self.epsilon:
            move =  max(moves, key=lambda x: x[0])
        return (move, reward)

    # Get an action according to the policy and updates the previous state it was in with the last action it took
    def sarsa(self):
        statePrime = abstractState(self) #s'
        moveAndReward = self.policy()   #a', r
        actionPrime = moveAndReward[0]  #a'
        reward = moveAndReward[1]       #r
        stateActionPrime = (abstractState(self), actionPrime[1]) #s', a'
        stateAction = (statePrime, self.lastAction) #s, a
        # we store the last action
        self.lastAction = actionPrime[1]  
        # only update the state when its training, while testing the action doesnt update since epsilon is 0
        if self.isTraining:
            qsaPrime = initialStateValue
            try:
                qsaPrime = qsa[stateActionPrime]    #q(s',a')
            except:
                qsa[stateActionPrime] = initialStateValue
            try:
                qsa[stateAction] = qsa[stateAction] + 0.05*(reward + 0.9*qsaPrime - qsa[stateAction])
                # q(s,a)         = q(s,a)           + alpha*[  r   + gamma*q(s',a') - q(s,a)]
            except:
                qsa[stateAction] = initialStateValue

#
#  4 functions we did not adjust
#
    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i ==0:
                c.draw(surface, True)
            else:
                c.draw(surface)    


def drawGrid(w, rows, surface):
    sizeBtwn = w // rows
 
    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn
 
        pygame.draw.line(surface, (255,255,255), (x,0),(x,w))
        pygame.draw.line(surface, (255,255,255), (0,y),(w,y)) 

def redrawWindow(surface):
    global s
    surface.fill((0,0,0))
    s.draw(surface)
    s.snack.draw(surface)
    drawGrid(w, rows, surface)
    pygame.display.update()

def message_box(subject, content):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    messagebox.showinfo(subject, content)
    try:
        root.destroy()
    except:
        pass

# end of functions we did not adjust

# reset the game
def gameOver():
    s.reset((startx, starty))

# check if the snake is colliding or head is out of the field
def isGameOver(s):
    if len(s.body) != len(s.locations):
        return True
    if ((s.body[0].pos[0] < 0)
    or (s.body[0].pos[0] > rows-1)
    or (s.body[0].pos[1] > rows-1)
    or (s.body[0].pos[1] < 0)):
        return True
    return False


def main():
    global  rows, s
    win = pygame.display.set_mode((w, w))
    s = Snake((255,0,0), (startx, starty))
    s.snack = Cube(s.randomSnack(rows), color=(0,255,0))
    flag = True
    games = 1
    clock = pygame.time.Clock()
    redrawWindow(win)
    while True:
        #training
        for i in range(1001):
            while flag:
                s.isTraining = True
                s.epsilon = 0.1
                clock.tick(4000)
                s.keyInput()
                s.sarsa()
                if s.dead:
                    gameOver()
                    games += 1
                    if games % 1000 == 0:
                        print("Game# ", games)
                    s.dead = False
                    break
                # every thousand games we test the agent
                if i  == 1000:
                    i = 0
                    tests = 1
                    averageScore = 0
                    
                   
                    for j in range(10): #testing
                        k = 0
                        while True:
                            s.isTraining = False
                            s.epsilon = 0.0
                            clock.tick(20)
                            s.keyInput()
                            s.sarsa()
                            if s.ate:
                                k = 0
                                s.ate = False                    
                            if s.dead or k == 199: # agent is terminated if it hasn't eaten food in 200 moves
                                s.dead = False
                                averageScore = (averageScore*(tests -1) + len(s.body))/(tests)
                                print("test# ", tests, " Score: ", len(s.body), "Average score:", averageScore)
                                s.snack = Cube(s.randomSnack(rows), color=(0,255,0))
                                gameOver()
                                tests += 1
                                break
                            k += 1
                            redrawWindow(win)
    pass
main()
