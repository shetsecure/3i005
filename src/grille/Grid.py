from Ship import Ship
from Direction import Direction
from Square import Square
from Allowed_ship import allowed_ships_dict
from copy import deepcopy

import itertools
import numpy as np
import matplotlib.pyplot as plt

class Grid():
    def __init__(self, lines=10, cols=10):
        assert(int(lines) > 0 and int(cols) > 0)

        self.lines = lines
        self.cols = cols
        self.grille = [[Square.empty] * cols for _ in range(lines)]
        self.matrix = np.zeros((lines, cols), dtype=int) # 0 means empty (will be used for plots)
        self.ships = list(deepcopy(allowed_ships_dict).values())
        self.already_hitted_places = []
        self.already_missed_places = []

    def reset(self):
        self.grille = [[Square.empty] * self.cols for _ in range(self.lines)]
        self.matrix = np.zeros((self.lines, self.cols), dtype=int) # 0 means empty (will be used for plots)
        self.ships = list(deepcopy(allowed_ships_dict).values())
        self.already_hitted_places = []
        self.already_missed_places = []

    def _inRange(self, x, y):
        return 0 <= x < self.lines and 0 <= y < self.cols
    
    # Override
    def __getitem__(self, coord):
        x, y = coord

        if not self._inRange(x, y):
            raise IndexError
        
        return self.grille[x][y]

    # Override
    def __setitem__(self, coord, state):
        x, y = coord
        
        if not self._inRange(x, y):
            raise IndexError
        
        all_possible_states = set(state.value for state in Square)

        if not (state.value in all_possible_states):
            raise ValueError(str(state) + " is not in Square(Enum)")

        self.grille[x][y] = state

    def can_place_ship(self, ship, coord, direction, place_the_ship = False):
        assert isinstance(ship, Ship)
        assert len(coord) == 2
        assert isinstance(direction, Direction)
        
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
            if any(self[x, y] != Square.empty for x, y in coordinates): # something is on the way
                return False

        except IndexError: # debordement (we're outside the grid)
            return False

        if place_the_ship:
            for x, y in coordinates:
                self.grille[x][y] = Square.ship
                self.matrix[x, y] = ship.color
            
            ship.coord = coord
            ship.direction = direction
            
        return True 

    def remove_ship(self, ship):
        assert ship in self.ships
        assert ship.coord != None and ship.direction != None

        directions = {
            'horizontal': (0, 1),
            'vertical': (1, 0),
        }
        dx, dy = directions[ship.direction.value]
        x, y = ship.coord

        coordinates = [
            (x + i * dx, y + i *dy)
            for i in range(ship.length)
        ]

        for x, y in coordinates:
            self.grille[x][y] = Square.empty
            self.matrix[x, y] = 0

    def place_ship(self, ship, coord, direction):
        # assertion will be done by can_place_ship
        if self.can_place_ship(ship, coord, direction, place_the_ship = True):
            if not (ship in self.ships):
                self.ships.append(ship)

            return True
        
        return False

    def place_ship_randomly(self, ship):
        directions = [_ for _ in Direction]
        random_direction = directions[np.random.randint(2)]
        positions = list(itertools.product(range(self.lines), range(self.cols)))

        random_index = np.random.randint(len(positions))

        placed = self.place_ship(ship, positions[random_index], random_direction)
        del positions[random_index]

        while not placed:
            random_direction = directions[np.random.randint(2)]
            random_index = np.random.randint(len(positions))

            placed = self.place_ship(ship, positions[random_index], random_direction)

            del positions[random_index]

    def generate_random_grid(self):
        for ship in self.ships:
            self.place_ship_randomly(ship)

    def get_placed_ships(self):
        placed = []

        for ship in self.ships:
            if ship.coord is not None:
                placed.append(ship)

        return placed
    
    def eq(self, other):
        assert(isinstance(other, Grid))
        same_dimension = self.lines == other.lines and self.cols == other.cols

        if not same_dimension:
            return False

        # no need to sort them, because the 2 grids will have the same order in the ships list
        # by construction
        for i in range(len(self.ships)):
            if not self.ships[i] == other.ships[i]:
                return False
        
        return True

    def is_empty(self):
        for ship in self.ships:
            if not ship.is_placed():
                return False
        
        return True

    def copy(self):
        # return a deep copy of the grid
        copy_grid = Grid(self.lines, self.cols)
        copy_grid.matrix = np.copy(self.matrix)
        copy_grid.grille = deepcopy(self.grille)
        copy_grid.ships = deepcopy(self.ships)
        copy_grid.already_hitted_places = deepcopy(self.already_hitted_places)

        return copy_grid

    def show(self):
        plt.imshow(self.matrix)
        #plt.set_cmap('hot')


if __name__ == '__main__':
    grid = Grid(10, 10)
    grid.generate_random_grid()
    grid.show()
    