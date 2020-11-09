""""

Includes classes for the agent and his knowledge base.  Some useful methods
are also in this file.


# FACING KEY:
#    0 = up
#    1 = right
#    2 = down
#    3 = left
"""

from knowledge_base import expr, dpll_satisfiable
from random import randint


def get_neighbors(r, c):
    #    Returns a list of the neighbors of position (r, c).
    neighbors = set()
    if (r + 1) in range(4):
        neighbors.add((r + 1, c))
    if (r - 1) in range(4):
        neighbors.add((r - 1, c))
    if (c + 1) in range(4):
        neighbors.add((r, c + 1))
    if (c - 1) in range(4):
        neighbors.add((r, c - 1))
    return neighbors


class WWKB():

    def __init__(self):
        # Knows there is no pit or wumpus in the starting spot
        self.knows = expr('~P30 & ~W30')

        # Lists of initial sentences
        templist = []
        for x in range(4):
            for y in range(4):
                for l, r in (('B', 'P'), ('S', 'W')):
                    tempLeft = "%s%s%s" % (l, x, y)
                    tempRight = []
                    for s, t in get_neighbors(x, y):
                        tempRight.append("%s%s%s" % (r, s, t))
                    templist.append("(%s <=> (%s))" % \
                                    (tempLeft, ' | '.join(tempRight)))
        implications = expr(' & '.join(templist))
        templist2 = ['W%s%s' % (i, j) for i in range(4) for j in range(4)]
        wumpus = expr(' | '.join(templist2))
        templist3 = ['(~W%s%s | ~W%s%s)' % (i, j, x, y)
                     for i in range(4) \
                     for j in range(4) \
                     for x in range(4) \
                     for y in range(4) \
                     if not ((i == x) and (j == y))]
        wumpus2 = expr(' & '.join(templist3))
        # Insert lists of sentence expressions into KB
        self.knows &= implications
        self.knows &= wumpus
        self.knows &= wumpus2

    def tell(self, sentence):
        self.knows &= sentence

    def ask(self, sentence):
        satisfiable = dpll_satisfiable(self.knows & ~sentence)
        if (satisfiable):
            return False
        else:
            return True


class WWAgent:

    def __init__(self):
        self.position = (3, 0)
        self.lastPos = (3, 0)
        self.facing = 1
        self.kb = WWKB()
        self.visited = set()
        self.lastAction = None
        self.plan = ['move']
        self.safe = set()
        self.notsafe = set()
        self.unsure = set()
        self.hasGold = False
        self.wumpusAlive = True
        self.escape = False
        self.arrow = 1
        self.start = True
        self.unsureGoal = False
        self.percepts = (None, None, None, None, None)
        self.knownWorld = [[[0 for x in range(5)] for x in range(4)] for x in range(4)]
        # print("New agent created")

    def update(self, percept):
        self.percepts = percept
        # [stench, breeze, glitter, bump, scream]

    def update_stats(self, pastAction):
        if (self.percepts[4] == 'scream'):
            self.wumpusAlive = False
        if (pastAction == 'right'):
            self.facing = (self.facing + 1) % 4
        elif (pastAction == 'left'):
            if (self.facing == 0):
                self.facing = 3
            else:
                self.facing = (self.facing - 1)
        # ASSUMING AGENT ONLY GRABS IF ON GOLD
        elif (pastAction == 'grab'):
            self.hasGold = True
        elif (pastAction == 'shoot'):
            self.arrow = 0
        elif (pastAction == 'move'):
            # Update position
            p = self.position
            if (self.facing == 0) and ((p[0] - 1) >= 0):
                self.lastPos = self.position
                self.position = (p[0] - 1, p[1])
            elif (self.facing == 1) and ((p[1] + 1) < 4):
                self.lastPos = self.position
                self.position = (p[0], p[1] + 1)
            elif (self.facing == 2) and ((p[0] + 1) < 4):
                self.lastPos = self.position
                self.position = (p[0] + 1, p[1])
            elif (self.facing == 3) and ((p[1] - 1) >= 0):
                self.lastPos = self.position
                self.position = (p[0], p[1] - 1)
        # print 'P-position: ', self.position

    def create_plan(self, goal):
        vis = {}
        pos = self.position
        currPos = self.position
        currFace = self.facing
        path = []
        plan = []

        for x in self.visited:
            vis[x] = 256
        vis[goal] = 0

        def neighbors(x, y, n):
            temp = []
            for i in get_neighbors(x, y):
                if (i in self.visited) and (i not in n):
                    temp.append(i)
            return temp

        goalPath = [goal]
        while (len(goalPath) > 0):
            gp = goalPath.pop()
            for x in neighbors(gp[0], gp[1], goalPath):
                if vis[x] > vis[gp] + 1:
                    vis[x] = vis[gp] + 1
                    goalPath.append(x)
        dis = vis[pos]
        while dis > 1:
            for v in neighbors(pos[0], pos[1], goalPath):
                if (vis[v] < dis):
                    dis = dis - 1
                    path.append(v)
                    pos = v
                    break
        path.append(goal)
        # path created
        # build plan backwards
        for p in path:
            if (p[0] - currPos[0]) > 0:
                face = 2
            elif (p[0] - currPos[0]) < 0:
                face = 0
            else:
                if (p[1] - currPos[1]) > 0:
                    face = 1
                else:
                    face = 3
            dif = face - currFace
            # handle facing direction
            if dif != 0:
                numTurn = abs(dif)
                if abs(dif) > 2:
                    dif = dif * -1
                    numTurn = numTurn % 2
                for q in range(numTurn):
                    if dif < 0:
                        plan.append('left')
                    elif dif > 0:
                        plan.append('right')
                currFace = face
            plan.append('move')
            currPos = p
        # put plan in correct order
        plan.reverse()
        # print 'Plan: ', plan

        return plan

    def action(self):
        self.update_stats(self.lastAction)
        per = self.percepts
        # Update knowledge base
        if self.position not in self.visited:
            # print('update kb')
            self.visited.add(self.position)
            pos = self.position
            lpos = self.lastPos
            # print(pos)
            self.knownWorld[pos[0]][pos[1]] = self.percepts
            if (self.percepts[0] == 'stench'):
                toTell = 'S%s%s' % (pos[0], pos[1])
                self.kb.tell(expr(toTell))
            else:
                toTell = '~S%s%s' % (pos[0], pos[1])
                self.kb.tell(expr(toTell))
            if (self.percepts[1] == 'breeze'):
                toTell = 'B%s%s' % (pos[0], pos[1])
                self.kb.tell(expr(toTell))
            else:
                toTell = '~B%s%s' % (pos[0], pos[1])
                self.kb.tell(expr(toTell))
            if (self.percepts[4] == 'scream'):
                self.wumpusAlive = False
            toTell1 = '~W%s%s' % (pos[0], pos[1])
            toTell2 = '~P%s%s' % (pos[0], pos[1])
            self.kb.tell(expr(toTell1))
            self.kb.tell(expr(toTell2))

            # Using corner logic
            # UPDATE SURROUNDINGS
            temp = get_neighbors(pos[0], pos[1]) - self.visited
            # print "surrounding-visited: ", temp
            temp = temp - self.safe
            temp = temp - self.notsafe
            tempS = set()
            tempN = set()
            if (per[0] is not 'stench') and (per[1] is not 'breeze'):
                self.unsure |= self.unsure - set((pos[0], pos[1]))
                tempS |= temp
            else:
                self.unsure |= temp
            tempF = self.unsure.copy()
            if len(self.visited) > 1:
                # add unsure locations as either safe or unsafe
                for f in tempF:
                    if self.kb.ask(expr('(P%s%s)' % (f[0], f[1]))):
                        self.unsure.remove(f)
                        tempN.add(f)
                    elif self.kb.ask(expr('(W%s%s)' % (f[0], f[1]))):
                        if self.wumpusAlive is True:
                            self.unsure.remove(f)
                            tempN.add(f)
                        else:
                            self.unsure.remove(f)
                            tempS.add(f)
                    elif self.kb.ask(expr('(~W%s%s & ~P%s%s)' % (f[0], f[1], f[0], f[1]))):
                        self.unsure.remove(f)
                        tempS.add(f)
            if len(tempS) > 0:
                self.safe |= tempS
            if len(tempN) > 0:
                self.notsafe |= tempN
            self.safe = self.safe - self.visited

        if (self.start is True) and (self.percepts[0] == 'stench'):
            action = 'shoot'
        elif (self.percepts[2] == 'glitter') and (self.hasGold is not True):
            action = 'grab'
        elif (self.position == (3, 0)) and ((self.hasGold is True) or (self.escape is True)):
            action = 'climb'
        elif (len(self.plan) > 0):
            # TAKE ACTION FROM PLAN
            action = self.plan.pop()
            # print "plan:action: ", action
        else:
            # CREATE PLAN
            # If the agent has gold his next goal should be the exit (start square)
            posGoals = None
            if self.hasGold is True:
                tempGoal = (3, 0)
                self.unsureGoal = False
            else:
                if len(self.safe) > 0:
                    posGoals = self.safe
                    self.unsureGoal = False
                elif len(self.unsure) > 0:
                    posGoals = self.unsure
                    self.unsureGoal = True
                else:
                    self.escape = True
                    self.unsureGoal = False
            if (posGoals is not None):
                tempGoal = posGoals.pop()
            else:
                tempGoal = (3, 0)
                self.unsureGoal = False

            self.plan = self.create_plan(tempGoal)

            # Shoot arrow before final move to make sure
            if (self.unsureGoal is True) and (len(self.plan) == 1) and \
                    (self.percepts[0] == 'stench') and (self.arrow == 1):
                action = 'shoot'
            else:
                action = self.plan.pop()

        self.start = False
        self.lastAction = action
        return action
