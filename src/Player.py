from Bataille import Bataille
from Square import Square
from Direction import Direction
from Allowed_ship import allowed_ships_dict
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
        """
            Method that returns the valid adjacent positions of (x, y)
        """
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

    def plot_expected_value(self, n):
        """"
            This method does essentialy 2 things:
                - Play random games n times
                - then it plots the distribution and prints the expected value.

            @params:
                n : int ( how many games the AI will play )
        """

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

            while not self.bataille.victory():
                count += self.play()

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
        # real definition here 
        # print("Expected value 2 : ", end='')
        # print(sum([(i+1)*j for i,j in zip(list(range(len(y))), y)]))
        # print(sum(y))
            
        total = 0
        for i in range(len(y)):
            total += y[i]
            z.append(total)
            
        plt.xlabel('After x hits')
        plt.ylabel('Probability to finish the game')
        
        plt.plot(x, z)
        txt = 'Expected value : ' + str(s)

        plt.title("Distribution de la variable aléatoire\n" + txt)
        plt.show()
    
        
class RandomPlayer(Player):
    """
        A player that plays randomly (randomly choose a position and shoot.)
        Each time he chooses a unique random position (never choose the same position twice.)
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
        A player that plays randomly (randomly choose a position and shoot.)
        Once he hit successfully a ship, he will shoot at the surroundings recursively.
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
    def plot_expected_value(self, n):
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

            while not self.bataille.victory():
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
        total = 0
        for i in range(len(y)):
            total += y[i]
            z.append(total)
            
        plt.xlabel('After x hits')
        plt.ylabel('Probability to finish the game')
        
        plt.plot(x, z)
        txt = 'Expected value : ' + str(s)

        plt.title("Distribution de la variable aléatoire\n" + txt)
        plt.show()
    
class ProbabilisticPlayer(Player):
    """
        A player the calculates each iteration the probability map of the game.
        Then he shoots at the square whith the highest probability.
        
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
        #self.bataille_copy = bataille.copy()
        #self.grid_copy = self.bataille_copy.grid.copy() # probabilities will be calculated in an empty grid
        #self.grid_copy.reset() # reset to make the grid empty (remove all ships)
        self.probabilities = self.recalculateProbabilities()
        
    def plotProbas(self, annotate=True):
        if annotate:
            for (j,i),label in np.ndenumerate(self.probabilities):
                plt.text(i,j,int(label),ha='center',va='center')
            
        plt.set_cmap('jet')
        plt.imshow(self.probabilities)

    def can_place_ship(self, ship, coord, direction):
        """
            We are using this method instead of the grid one, because the grid is initially full.
            Some positions are filled, so rather than creating another empty grid, we just change 
            the cretiria of placing a ship.
            A ship can be placed if there is enough room ( enough empty or MISSED squares )
        """
        directions = {
            'horizontal': (0, 1),
            'vertical': (1, 0),
        }
        dx, dy = directions[direction.value]
        x, y = coord

        coordinates = [
            (x + i * dx, y + i *dy)
            for i in range(ship.length)
        ]
        
        try:
            if any(self.grid[x, y] == Square.missed for x, y in coordinates): 
                return False

        except IndexError: # debordement (we're outside the grid)
            return False

        return True

    def recalculateProbabilities(self):
        probas = np.zeros((self.grid.lines, self.grid.cols))
        
        for ship in self.grid.ships:
            for x in range(self.grid.lines):
                for y in range(self.grid.cols):
                    if self.can_place_ship(ship, (x, y), Direction.horizontal):
                        probas[x, y : y+ship.length] += 1

                    if self.can_place_ship(ship, (x, y), Direction.vertical):
                        probas[x : x+ship.length, y] += 1
        
        # making the surroundings of hitted places more probable
        hits = self.grid.already_hitted_places
        surr = []

        for hit in hits:
            x, y = hit
            surr.append(self.connex(x, y))

        surr = set(itertools.chain.from_iterable(surr))
        
        for x, y in surr:
            probas[x, y] *= 2

        # already hitted places, shouldn't be hitted again
        for x, y in hits:
            probas[x, y] = 0

        return probas
        
    def play(self):
        # get the best position (position with the highest probability)
        x, y = np.unravel_index(np.argmax(self.probabilities), self.probabilities.shape)

        self.bataille.play(x, y)
        
        self.probabilities = self.recalculateProbabilities()

        return 1

    def trace_game(self, pause_rate=1):
        """
            This method for tracing a random game, it will show the positions of the ships, then 
            the probability map before each hit.

            @params:
                - pause_rate: how much the plot will pause in seconds.
        """
        count = 0
        b = self.bataille

        while not b.victory():
            b.show()
            plt.draw()
            plt.pause(pause_rate)
            plt.clf()
            p.plotProbas()
            count += p.play()
            plt.draw()
            plt.pause(pause_rate)
            plt.clf()

        print(count)