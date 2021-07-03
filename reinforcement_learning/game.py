import numpy as np

from constants import *
from utils import *


class Teeko:
    def __init__(self):
        self.black_grid = np.zeros((5,5))
        self.red_grid = np.zeros((5,5))
        
        self.black_placed = 0
        self.red_placed = 0

        self.turn_ind = 0
        self.turn = TURNS[self.turn_ind]
        self.phase = "placing"
        self.winner = None
        self.game_over = False
        self.game_length = 0

        self.move_mask = self.legal_move_mask()


    
    def act(self, action):  # action MUST be a legal move

        if self.move_mask[action] == 0:
            return

        if self.phase == "placing":

            # action is in [0, 24]
            x, y = coord_from_int(action)

            if self.turn == "black":

                self.black_grid[x][y] = 1
                self.black_placed += 1

            elif self.turn == "red":

                self.red_grid[x][y] = 1
                self.red_placed += 1

        elif self.phase == "moving":

            # action is in [0, 199]  (200 = 25 * 8)
            direction = action % 8
            pos = int((action - direction)/8)

            x, y = coord_from_int(pos)

            x_new = x + DIRECTIONS[direction][0]
            y_new = y + DIRECTIONS[direction][1]

            if self.turn == "black":

                self.black_grid[x][y] = 0
                self.black_grid[x_new][y_new] = 1

            elif self.turn == "red":

                self.red_grid[x][y] = 0
                self.red_grid[x_new][y_new] = 1

        
        self.game_length += 1
        
        self.check_win()

        if self.game_over:
            return

        self.pass_turn()
        self.check_phase()
        self.move_mask = self.legal_move_mask()

    
    def pass_turn(self):
        self.turn_ind = (self.turn_ind + 1) % 2
        self.turn = TURNS[self.turn_ind]

    
    def check_phase(self):
        if self.black_placed == 4 and self.red_placed == 4:
            self.phase = "moving"

    
    def legal_move_mask(self):
        #moves = []
        mask = []

        side = self.turn

        if self.phase == "placing":
            mask = [0 for _ in range(25)]

            for i in range(25):

                x, y = COORDS[i]

                # can not place pieces on already occupied squares
                if self.red_grid[x][y] == 1 or self.black_grid[x][y] == 1:
                    continue

                #moves.append(i)
                mask[i] = 1

        elif self.phase == "moving":
            mask = [0 for _ in range(200)]

            for i in range(200):

                # can not move off the board
                if not EDGE_MASK[i]: 
                    continue
                
                direction = i % 8
                pos = int((i - direction)/8)

                x, y = COORDS[pos]

                # can not move a piece that isn't there
                if side == "black" and self.black_grid[x][y] == 0:
                    continue

                elif side == "red" and self.red_grid[x][y] == 0:
                    continue

                x_new = x + DIRECTIONS[direction][0]
                y_new = y + DIRECTIONS[direction][1]

                # can not move where another piece already is
                if self.red_grid[x_new][y_new] == 1 or self.black_grid[x_new][y_new] == 1:
                    continue

                #moves.append(i)
                mask[i] = 1

        return mask

    
    def check_win(self):
        grid = []
        strides = [(1, 0), (1, 1), (0, 1), (-1, 1)]

        side = self.turn

        if side == "red":
            grid = self.red_grid

        elif side == "black":
            grid = self.black_grid

        for x in range(5):
            for y in range(5):

                if grid[x][y] == 1:
                    for stride in strides:
                        if is_winning_streak(grid, (x, y), stride):
                            self.winner = side
                            self.game_over = True

                    return None

    def represent(self):
        if self.turn_ind == 0:
            turn_plane = np.zeros((1, 5, 5))
        elif self.turn_ind == 1:
            turn_plane = np.ones((1, 5, 5))

        black_plane = np.expand_dims(self.black_grid, axis=0)
        red_plane = np.expand_dims(self.red_grid, axis=0)

        output = np.concatenate((turn_plane, black_plane, red_plane))

        return output