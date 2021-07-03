from utils import draw_from


class Node:
    def __init__(self, board, action):
        self.board = board
        self.action = action

        self.value = None  # total value of next state
        self.visit_count = None 
        self.Q = None  # mean value of next state
        self.p = None  # probability of selecting action