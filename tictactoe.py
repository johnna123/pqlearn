import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np

from coms import cls
from qlearn import QLearn

"""
This code is not working yet
"""


class TttGame:
    def __init__(self, verbose=False, name="ttt_ai.pkl"):
        self.board = np.zeros((3, 3))
        self.board_pieces = ['   ', ' X ', ' O ', ' ? ']
        self.board_coords = {"ul": (0, 0), "u": (0, 1), "ur": (0, 2), "l": (1, 0), "c": (1, 1), "r": (1, 2),
                             "dl": (2, 0), "d": (2, 1), "dr": (2, 2)}
        self.winner = 0
        self.end = False
        self.ai_name = name
        self.ai = QLearn(list(self.board_coords.keys()), no_repeat=True, epsilon=0.5)
        if self.ai_name in os.listdir():
            self.ai = joblib.load(self.ai_name)
        self.ai.verbose = verbose
        self.ai.epsilon = 0

    def box_select(self, player, ai_controlled=False):
        if ai_controlled:
            select = self.ai.interact([self.board, player])
        else:
            select = input("player {} selection: ".format(player))
        if not self.board[self.board_coords[select]]:
            self.board[self.board_coords[select]] = player
        else:
            self.board[self.board_coords[select]] = 3
        return select

    def winner_check(self):
        for i in range(3):
            if self.board[i, 0] == self.board[i, 1] == self.board[i, 2] != 0:
                self.make_winner((i, 0))
            if self.board[0, i] == self.board[1, i] == self.board[2, i] != 0:
                self.make_winner((0, i))
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0 or self.board[0, 2] == self.board[1, 1] == \
                self.board[2, 0] != 0:
            self.make_winner((1, 1))

    def make_winner(self, win):
        self.end = True
        self.winner = self.board[win]

    def check_events(self):
        if np.max(self.board) == 3:
            self.end = True
        else:
            self.winner_check()
            if np.min(self.board) != 0:
                self.end = True

    def print_board(self):
        cls()
        print('{}|{}|{}'.format(self.board_pieces[int(self.board[0][0])], self.board_pieces[int(self.board[0][1])],
                                self.board_pieces[int(self.board[0][2])]))
        for i in range(1, 3):
            print("-" * 11)
            print('{}|{}|{}'.format(self.board_pieces[int(self.board[i][0])], self.board_pieces[int(self.board[i][1])],
                                    self.board_pieces[int(self.board[i][2])]))

    def reward(self, player, policy):
        pr = {"ul": .6, "u": .3, "ur": .6, "l": .3, "c": 1, "r": .3, "dl": .6, "d": .3, "dr": .6}
        if self.winner == player:
            return 2
        if self.end and self.winner == 0:
            return -2
        return pr[policy]

    def turn(self, player, ai_controlled=False, light_mode=False):
        old_state = [self.board.copy(), player]
        policy = self.box_select(player, ai_controlled=ai_controlled)
        self.check_events()
        self.ai.update_q(old_state, [self.board, player], self.reward(player, policy), policy)
        if not light_mode:
            self.print_board()

    def play(self):
        self.print_board()
        while True:
            self.turn(1)
            if self.end:
                self.ai.save_model(self.ai_name)
                return self.winner
            self.turn(2)
            if self.end:
                self.ai.save_model(self.ai_name)
                return self.winner

    def play_mvm(self, light_mode=False):
        if not light_mode:
            self.print_board()
        while True:
            self.turn(1, ai_controlled=True, light_mode=light_mode)
            if not light_mode:
                time.sleep(0.5)
            if self.end:
                self.ai.save_model(self.ai_name)
                return self.winner
            self.turn(2, ai_controlled=True, light_mode=light_mode)
            if not light_mode:
                time.sleep(0.5)
            if self.end:
                self.ai.save_model(self.ai_name)
                return self.winner


def auto_train(iters):
    evals = list()
    for _ in range(iters):
        game = TttGame()
        # game.ai.no_repeat = True
        game.ai.epsilon = 1
        game.play_mvm(light_mode=True)
        evals.append(np.sum(np.array([v for v in game.ai.memory.values()])))
        plt.plot(evals, c="b")
        plt.pause(0.05)


if __name__ == '__main__':
    # auto_train(100)
    game = TttGame(verbose=True)
    print(game.play())
    print("fin")
