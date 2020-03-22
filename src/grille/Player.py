from Bataille import Bataille
from Square import Square
from Direction import Direction
import itertools

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractclassmethod

class Player(ABC):
    def __init__(self, bataille):
        assert isinstance(bataille, Bataille)
        
        self.bataille = bataille
        self.grid = self.bataille.grid
        self.positions = list(itertools.product(range(self.grid.lines), range(self.grid.cols) ))

    def reset(self):
        self.positions = list(itertools.product(range(self.grid.lines), range(self.grid.cols) ))

    def connex(self, x, y):
        # up, down, left, right
        coords = ((x-1, y), (x+1, y), (x, y-1), (x, y+1)) 
        connex = []

        for x, y in coords:
            if self.grid._inRange(x, y) and (x, y) in self.positions:
                connex.append((x, y))
        
        return connex

    @abstractclassmethod
    def play(self):
        raise NotImplementedError

    def plot(self, n):
        assert(int(n) > 0)

        x = list(range(self.grid.cols * self.grid.lines))
        y = [0] * (self.grid.cols * self.grid.lines) 
        counts = [] # to get the expected value
        z = [] # accumulative probability for density function
        minimum = self.grid.cols * self.grid.lines
        s = 0 # sum
        
        for i in range(n):
            count = 0
            self.bataille.reset()
            self.reset()

            while not b.victory():
                count += self.play()

            print(count)
            y[count-1] += 1
            counts.append(count)
            
            if count < minimum:
                minimum = count
                
            s += count
        
        s /= n
        s = int(s)
        print("Best case (minimum) : ", minimum)
        print("Expected value : ", s)
        
        y = np.asarray(y) / n
        #print("Expected value 2 : ")
        # real definition here 
        #print(sum([(i+1)*j for i,j in zip(list(range(len(y))), y)]))
        #print(sum(y))
            
        total = 0
        for i in range(len(y)):
            total += y[i]
            z.append(total)
            
        plt.xlabel('Iteration (Game number)')
        plt.ylabel('Number of hits')
        
        plt.plot(x, z)

        plt.title("Distribution de la variable aléatoire")
        plt.show()
    
        
class RandomPlayer(Player):
    """
        A player that plays randomly (randomly choose a position and hit.)
        Each time he chooses a random position (never choose the same position twice.)
    """
    
    def __init__(self, bataille):
        super().__init__(bataille)
        
    def play(self):
        random_index = np.random.randint(len(self.positions))
        x, y = self.positions[random_index]
        self.bataille.play(x, y)
        
        del self.positions[random_index]
        
        return 1 # hitted 1 place

class HeuristicPlayer(Player):
    """
        A player that plays randomly (randomly choose a position and hit.)
        Each time he chooses a random position (never choose the same position twice.)
    """
    def __init__(self, bataille):
        super().__init__(bataille)
            
    def play(self, count_hits):
        random_index = np.random.randint(len(self.positions))
        x, y = self.positions[random_index]

        count_hits += 1

        if self.bataille.play(x, y): # hitted a target
            surr = self.connex(x, y) # get surroundings 

            for pos in surr: # hit connex regions
                xx, yy = pos
                if self.bataille.play(xx, yy):
                    count_hits = self.play(count_hits) # hit it recursively 
                
            for pos in surr:
                if pos in self.positions: 
                    # remove those positions from the list so, we don't hit them twice
                    self.positions.remove(pos)

        if (x, y) in self.positions: 
            # remove the random picked position from the list
            self.positions.remove((x, y))

        return count_hits

    # need to ovverride this so I can call play with 0 as argument.
    def plot(self, n):
        assert(int(n) > 0)

        x = list(range(self.grid.cols * self.grid.lines))
        y = [0] * (self.grid.cols * self.grid.lines) 
        counts = [] # to get the expected value
        z = [] # accumulative probability for density function
        minimum = self.grid.cols * self.grid.lines
        s = 0 # sum
        
        for i in range(n):
            count = 0
            self.bataille.reset()
            self.reset()

            while not b.victory():
                count += self.play(0)

            y[count-1] += 1
            counts.append(count)
            
            if count < minimum:
                minimum = count
                
            s += count
        
        s /= n
        s = int(s)
        print("Best case (minimum) : ", minimum)
        print("Expected value : ", s)
        
        y = np.asarray(y) / n
        #print("Expected value 2 : ")
        # real definition here 
        #print(sum([(i+1)*j for i,j in zip(list(range(len(y))), y)]))
        #print(sum(y))
            
        total = 0
        for i in range(len(y)):
            total += y[i]
            z.append(total)
            
        plt.xlabel('Iteration (Game number)')
        plt.ylabel('Number of hits')
        
        plt.plot(x, z)

        plt.title("Distribution de la variable aléatoire")
        plt.show()
    

class ProbabilisticPlayerOld(Player):
    """
        A player the calculates each iteration the probability map of the game.
        
        Extrait:
        À chaque tour, pour chaque bateau restant, on calcule la probabilité pour chaque case 
        de contenir ce bateau sans tenir compte de la position des autres bateaux.
        Pour cela, en examinant toutes les positions possibles du bateau sur la grille, pour chaque 
        case on obtient le nombre de fois où le bateau apparaît potentiellement.
        On dérive ainsila probabilité jointe de la présence d’un bateau sur une case
        (en considérant que la position des bateaux est indépendante).
    """

    def __init__(self, bataille):
        super().__init__(bataille)
        self.grid_copy = self.grid.copy()
        self.grid_copy.reset() # reset to make the grid empty (remove all ships)
        self.probas = self.calculate_probas()
        self.allowed_successif_misses = 3
        self.misses = self.allowed_successif_misses
    
    def plotProbas(self, annotate=True):
        if annotate:
            for (j,i),label in np.ndenumerate(self.probas):
                plt.text(i,j,int(label),ha='center',va='center')
            
        plt.set_cmap('jet')
        plt.imshow(self.probas)

    def calculate_probas(self): 
        """
            Probabilities of searching mode (we didn't hit anything yet).
            Pour cela, en examinant toutes les positions possibles du bateau sur la grille, 
            pour chaque case on obtient le nombre de fois où le bateau apparaît potentiellement.
        """
        probas = np.zeros((self.grid.lines, self.grid.cols)) # float by default

        # on calcule la probabilité pour chaque case 
        # de contenir ce bateau sans tenir compte de la position des autres bateaux.
        for ship in self.grid.ships:
            for x, y in self.positions:
                for direction in Direction:
                    if self.grid_copy.can_place_ship(ship, (x, y), direction):
                        if direction == Direction.horizontal:
                            probas[x, y: y+ship.length] += 1 # no worries if we overflowed
                        else:
                            probas[x: x+ship.length, y] += 1 # numpy will take care of it

        #probas /= np.max(probas)

        for hit in self.grid.already_hitted_places:
            x, y = hit
            probas = self.attack(x, y)
        
        for x, y in self.grid.already_hitted_places:
            probas[x, y] = 0

        for _x, _y in self.grid.already_missed_places:
            probas[_x, _y] = 0 

        return probas

    def attack(self, x, y):
        """
            We hitted something at (x, y) time to calculate different probabilites.
            For each ship we calculate how many ways we can put it touching (x, y).
        """
        prob = self.probas

        for ship in self.grid.ships:
            for direction in Direction:
                if direction == Direction.horizontal:
                    for i in range(y - ship.length + 1, y + 1):
                        if i >= 0:
                            if self.grid_copy.can_place_ship(ship, (x, i), direction):
                                prob[x, i: i + ship.length] += 1
                else:
                    for i in range(x - ship.length + 1, x + 1):
                        if i >= 0:
                            if self.grid_copy.can_place_ship(ship, (i, y), direction):
                                prob[i: i + ship.length, y] += 1

        for _x, _y in self.grid.already_hitted_places:
            prob[_x, _y] = 0 

        for _x, _y in self.grid.already_missed_places:
            prob[_x, _y] = 0 

        return prob


    def play(self):
        x, y = np.unravel_index(np.argmax(self.probas), self.probas.shape)

        if self.bataille.play(x, y):
            self.misses = 0
        else:
            self.misses += 1
            print('sucessive misses ', self.misses)

        self.probas = self.calculate_probas()

        
        if self.misses < self.allowed_successif_misses:
            for hit in self.grid.already_hitted_places:
                x, y = hit
                self.probas = self.attack(x, y)
            
            for x, y in self.grid.already_hitted_places:
                self.probas[x, y] = 0

            for _x, _y in self.grid.already_missed_places:
                self.probas[_x, _y] = 0 

            # attack mode
        if (x, y) in self.positions:
            self.positions.remove((x, y))

        return 1

class ProbabilisticPlayer(Player):
    def __init__(self, bataille):
        super().__init__(bataille)
        self.bataille_copy = bataille.copy()
        self.grid_copy = self.bataille_copy.grid.copy() # probabilities will be calculated in an empty grid
        self.grid_copy.reset() # reset to make the grid empty (remove all ships)
        self.probabilities = self.recalculateProbabilities()

    def recalculateProbabilities(self):
        hits = self.grid.already_hitted_places

        # reset probas
        probabilities = np.zeros((self.grid.lines, self.grid.cols))

        # calculate probas for each ship 
        for ship in self.grid.ships:
            for i in range(self.grid.lines):
                for j in range(self.grid.cols):
                    if self.grid_copy.can_place_ship(ship, (i, j), Direction.horizontal):
                        probabilities[i, j: j + ship.length] += 1

                    if self.grid_copy.can_place_ship(ship, (i, j), Direction.vertical):
                        probabilities[i: i + ship.length, j] += 1

        # skew probabilities for positions adjacent to hits
        print(hits)
        pos = []
        for s in hits:
            x, y = s
            pos.append(self.connex(x, y))

        pos = set(itertools.chain.from_iterable(pos))
        print(pos)

        for x, y in pos:
            probabilities[x, y] *= 2


        for x, y in self.grid.already_missed_places:
            probabilities[x, y] = 0

        for x, y in self.grid.already_hitted_places:
            probabilities[x, y] = 0

        return probabilities
        
    def plotProbas(self, annotate=True):
        if annotate:
            for (j,i),label in np.ndenumerate(self.probabilities):
                plt.text(i,j,int(label),ha='center',va='center')
            
        plt.set_cmap('jet')
        plt.imshow(self.probabilities)

    def getBestUnplayedPosition(self):
        matrix = self.bataille.grid.matrix
        bestProba = 0
        bestPos = (0, 0)

        for x in range(self.grid.lines):
            for y in range(self.grid.cols):
                if matrix[x, y] != -1 and self.probabilities[x, y] > bestProba:
                    bestProba = self.probabilities[x, y]
                    bestPos = (x, y)

        return bestPos

    def play(self):
        x, y = self.getBestUnplayedPosition()

        self.bataille.play(x, y)
        self.bataille_copy.play(x, y)

        if (x,y) in self.positions:
            index = self.positions.index((x,y))
            del self.positions[index]

        self.recalculateProbabilities()

        return 1
        
b = Bataille()

p = ProbabilisticPlayerOld(b)
p.plot(100)
# count = 0
# rate_pause = 0.001
# rate_pause = 1
# while not b.victory():
#     b.show()
#     plt.draw()
#     plt.pause(rate_pause)
#     plt.clf()
#     p.plotProbas()
#     count += p.play()
#     plt.draw()
#     plt.pause(rate_pause)
#     plt.clf()

# print(count)