from Direction import Direction

class Ship:
    # all attributes are public for shorter code ( for now... )

    def __init__(self, name, length, color, coord=None, direction=None):
        assert(isinstance(name, str))
        assert(int(color) > 0 and int(color) < 255)
        
        if coord is not None:
            assert(len(coord) == 2)
        
        if direction is not None:
            assert(isinstance(direction, Direction))
        
        self.name = name
        self.length = length
        self.coord = coord
        self.direction = direction
        self.color = color

    def reset(self):
        self.coord = None
        self.direction = None

    def __eq__(self, other):
        if isinstance(other, Ship):
            return (self.name == other.name and self.length == other.length \
                    and self.coord == other.coord and self.direction == other.direction \
                        and self.color == other.color )

        return False

    def is_placed(self):
        return self.coord == None and self.direction == None

    def __str__(self):
        s = "Ship name: " + self.name + '\n'
        s += "Ship color: " + str(self.color) + '\n'
        s += "Ship length: " + str(self.length) + '\n'

        if self.coord is not None:        
            s += "Ship coord: " + str(self.coord) + '\n'
            s += "Ship orientation: " + self.direction.value
        else:
            s += "Ship coord: None\n"
            s += "Ship orientation: None\n"

        return s
