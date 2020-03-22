from Grid import Grid
from Ship import Ship
from Direction import Direction
from copy import deepcopy
import sys

from helper import *


def how_many_ways_to_put_ship(lines, cols, ship_length):
    """ 
        Assumption: the grid is a squarred matrix
            @params: 
                - lines : int > 0
                - cols : int > 0
                - ship_length : int > 0

            returns: how many ways to put this ship in an empty grid(lines, cols)
    """

    assert int(lines) > 0 and int(cols) == int(lines) and int(ship_length) > 0

    if ship_length > 1:
        return (lines - ship_length + 1) * cols + (cols - ship_length + 1) * lines
    else:
        return (lines - ship_length + 1) * cols 

# fonction qui permet de dénombrer le nombre de façon de placer une liste de bateaux sur une grille vide. 
def how_many_conf(ships_lengths, lines, cols, visualize = False):
    """
        Fonction qui permet de dénombrer le nombre de façon de placer une liste de bateaux sur une grille vide. 
        
        Assumption: 
            placing all the ships is possible in a grid of the size lines * cols

        @params: 
            - ships_lengths: list of ints
            - lines : int
            - cols : int 
            - visualize : boolean to display the grid each iteration
    """
    assert int(lines) > 0 and int(cols) > 0
    assert len(ships_lengths) > 0
    for l in ships_lengths:
        assert int(l) > 0

    grid = Grid(lines, cols)
    ships = construct_ships(ships_lengths)
    count = calculate(grid, ships, lines, cols, 0, visualize)
    
    return count

def how_much_to_get_grid(grid):
    # fonction qui prend en paramètre une grille, génère des grilles aléatoirement jusqu’à ce que la grille 
    # générée soit égale à la grille passée en paramètre et qui renvoie le nombre de grilles générées.
    assert isinstance(grid, Grid)
    count = 1
    
    ships_copy = deepcopy(grid.ships)

    # generating new grid with same ships
    generated_grid = Grid(grid.lines, grid.cols)
    generated_grid.ships = ships_copy

    # reseting ships ( coords = direction = None )
    for i in range(len(generated_grid.ships)):
        generated_grid.ships[i].reset() 

    generated_grid.generate_random_grid()

    equals = grid.eq(generated_grid)

    while not equals:
        count += 1

        # generating new grid with same ships
        generated_grid = Grid(grid.lines, grid.cols)
        generated_grid.ships = ships_copy

        # reseting ships ( coords = direction = None )
        for i in range(len(generated_grid.ships)):
            generated_grid.ships[i].reset() 

        generated_grid.generate_random_grid()
        equals = grid.eq(generated_grid)
        
        if count % 3 == 0:
            sys.stdout.write("\r%d" % count)
            sys.stdout.flush()
    
    print(count)
    return count