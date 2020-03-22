from Grid import Grid
from Direction import Direction
from Square import Square

class Bataille:
    def __init__(self):
        self.grid = Grid()
        self.grid.generate_random_grid()
        self.required_hits_to_win = 0

        for ship in self.grid.ships:
            self.required_hits_to_win += ship.length
        
    def reset(self):
        self.grid.reset()
        self.grid.generate_random_grid()
        self.required_hits_to_win = 0

        for ship in self.grid.ships:
            self.required_hits_to_win += ship.length

    def show(self):
        self.grid.show()

    def play(self, x, y): # see what u will do for already hitted places
        grid = self.grid
        grid.matrix[x, y] = -1 # hit or missed
        

        if grid[x, y] == Square.ship:
            grid.already_hitted_places.append((x, y))
            grid[x, y] = Square.hitted
            self.required_hits_to_win -= 1
            return True
        
        grid.already_missed_places.append((x, y))
        grid[x, y] = Square.missed
        return False

    def victory(self):
        return self.required_hits_to_win == 0

    def copy(self):
        b_copy = Bataille()
        b_copy.grid = self.grid.copy()
        b_copy.required_hits_to_win = self.required_hits_to_win

        return b_copy