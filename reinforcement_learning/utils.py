import numpy as np

from constants import *



def generate_edge_mask():  # entries are False when move is impossible
    edge_mask = []

    for i in range(200):
        entry = True

        direction_ind = i % 8
        direction = DIRECTIONS[direction_ind]
        pos = int((i - direction_ind)/8)

        x = pos % 5
        y = int((pos - x)/5)

        x_new = x + direction[0]
        y_new = y + direction[1]

        if x_new < 0 or x_new > 4:
            entry = False
        elif y_new < 0 or y_new > 4:
            entry = False
        
        edge_mask.append(entry)

    return edge_mask


def coord_from_int(num):
    assert num < 25

    x = num % 5
    y = int((num - x)/5)

    return x, y


def int_from_coord(coord):  # coord: 2-len tuple or list 
    x, y = coord

    return 5*y + x


def int_from_move(move):
    x0 = move[0]
    x1 = move[1]

    a = int_from_coord(x0)

    d = (x1[0] - x0[0], x1[1] - x0[1])

    b = DIRECTIONS.index(d)

    return 8*a + b


def move_from_int(n):
    direction = n % 8
    pos = int((n - direction)/8)

    x, y = COORDS[pos]

    x_new = x + DIRECTIONS[direction][0]
    y_new = y + DIRECTIONS[direction][1]

    return ((x, y), (x_new, y_new))


def is_winning_streak(grid, pos, stride):  # both tuples
    x, y = pos
    m, n = stride

    if grid[x][y] == 0:
        return False
    
    x_bound = x + 3*m
    y_bound = y + 3*n

    if x_bound < 0 or x_bound > 4:
        return False
    if y_bound < 0 or y_bound > 4:
        return False

    for i in range(1,4):
        _x = x + i*m
        _y = y + i*n

        if grid[_x][_y] == 0:
            return False

    return True


def draw_from(cumdist): # draw int from cumulative distribution
    x = np.random.random()

    for i in range(cumdist.shape[0]):
        if x < cumdist[i]:
            return i
