from Grid import Grid
from Allowed_ship import allowed_ships_dict
from Bataille import Bataille
from Player import *
from combinatorics import *

if __name__ == '__main__':
    b = Bataille()
    p = HeuristicPlayer(b)
    p.plot_expected_value(100)