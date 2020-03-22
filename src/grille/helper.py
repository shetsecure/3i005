from Grid import Grid
from Ship import Ship
from Direction import Direction

import matplotlib.pyplot as plt

def construct_ships(ships_lengths):
    assert(len(ships_lengths) > 0)

    ships = []
    for length in ships_lengths:
        ships.append(Ship('temp_' + str(length), length, length))

    return ships

# used for how_many_conf(ships_length, lines, cols, visualize)
def calculate(grid, ships, lines, cols, count, visualize):
    if len(ships) == 1:
        for i in range(lines):
            for j in range(cols):
                for direction in Direction:
                    if grid.can_place_ship(ships[0], (i,j), direction):
                        if visualize:
                            grid.place_ship(ships[0], (i,j), direction)
                            grid.show()
                            plt.show()
                            grid.remove_ship(ships[0])
                            
                        count += 1
                            
    else:
        for i in range(lines):
            for j in range(cols):
                for direction in Direction:
                    if grid.can_place_ship(ships[0], (i,j), direction):
                        grid.place_ship(ships[0], (i,j), direction)
                        count = calculate(grid, ships[1:], lines, cols, count, visualize)

                        grid.remove_ship(ships[0])
                            
    return count
