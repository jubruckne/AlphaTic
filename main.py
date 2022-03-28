import random
import threading
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from termcolor import colored
from time import perf_counter

BOARD_COLS = 3
BOARD_ROWS = 3

PLAYER_1 = 1
PLAYER_2 = -1
TIE = 2
NONE = 0

SILENT = True


class Player:
    name = ""

    def __init__(self, name):
        self.name = name
        self.rnd = random.Random()
        return

    # input next move
    def make_move(self, tiles: ndarray, available: List[Tuple[int, int]], player_id):
        while True:
            print(self.name + " make your move...")
            row = int(input("Row:")) - 1
            col = int(input("Column:")) - 1
            action = (row, col)
            if action in available:
                return action

    def game_over(self, score):
        pass

    def reset(self):
        pass


class AILearning(Player):
    def __init__(self, name='Learning', exp_rate=0.12, learning_rate=0.1):
        self.states = []
        self.state_values = {}  # state -> value
        self.exp_rate = exp_rate
        self.learning_rate = learning_rate

        Player.__init__(self, name)

    def make_move(self, tiles, available, player_id):
        value_max = -999

        for p in self.rnd.sample(available, len(available)):
            next_tiles = tiles.copy()
            next_tiles[p] = player_id
            next_hash = str(next_tiles.reshape(BOARD_COLS * BOARD_ROWS))
            value = 0 if self.state_values.get(next_hash) is None else self.state_values.get(next_hash)
            if value > value_max:
                value_max = value
                action = p

        if value_max < 0.01:
            # pick randomly out of the possible moves
            action = self.rnd.choice(available)
            next_tiles = tiles.copy()
            next_tiles[action] = player_id
            next_hash = str(next_tiles.reshape(BOARD_COLS * BOARD_ROWS))
        else:
            if self.rnd.random() * value_max <= self.exp_rate :
                # pick randomly out of the possible moves
                action = self.rnd.choice(available)
                next_tiles = tiles.copy()
                next_tiles[action] = player_id
                next_hash = str(next_tiles.reshape(BOARD_COLS * BOARD_ROWS))

        if not SILENT:
            print("{0} is making it's move: {1}, {2}".format(self.name, action[0] + 1, action[1] + 1))

        self.states.append(next_hash)

        return action

    def game_over(self, score):
        # print(self.name + " score: " + str(score))
        # print(self.states)

        for st in reversed(self.states):
            if self.state_values.get(st) is None:
                self.state_values[st] = 0
            self.state_values[st] += self.learning_rate * (0.9 * score - self.state_values[st])

            score = self.state_values[st]

        # print(self.state_values)
        #self.exp_rate = self.exp_rate * 0.9999999
        #print("exp_rate = " + str(self.exp_rate))

    def reset(self):
        self.states = []

class AIRandomEnhanced(Player):
    def __init__(self, name='Enhanced'):
        Player.__init__(self, name)

    def make_move(self, tiles, available, player_id):
        if tiles[1, 1] == NONE:
            return 1, 1

        options = []

        for i in range(BOARD_ROWS):
            if tiles[i, 0] != NONE and tiles[i, 0] == tiles[i, 1] and tiles[i, 2] == NONE:
                options.append((i, 2))
            if tiles[i, 0] != NONE and tiles[i, 0] == tiles[i, 2] and tiles[i, 1] == NONE:
                options.append((i, 1))
            if tiles[i, 1] != NONE and tiles[i, 1] == tiles[i, 2] and tiles[i, 0] == NONE:
                options.append((i, 0))

        for i in range(BOARD_COLS):
            if tiles[0, i] != NONE and tiles[0, i] == tiles[1, i] and tiles[2, i] == NONE:
                options.append((2, i))
            if tiles[0, i] != NONE and tiles[0, i] == tiles[2, i] and tiles[1, i] == NONE:
                options.append((1, i))
            if tiles[1, i] != NONE and tiles[1, i] == tiles[2, i] and tiles[0, i] == NONE:
                options.append((0, i))

        options2 = []

        if tiles[0, 0] != NONE and tiles[0, 0] == tiles[1, 1] and tiles[2, 2] == NONE:
            options2.append((2, 2))

        if tiles[0, 0] != NONE and tiles[0, 0] == tiles[2, 2] and tiles[1, 1] == NONE:
            options2.append((1, 1))

        if tiles[1, 1] != NONE and tiles[1, 1] == tiles[2, 2] and tiles[0, 0] == NONE:
            options2.append((0, 0))

        if tiles[2, 0] != NONE and tiles[2, 0] == tiles[1, 1] and tiles[0, 2] == NONE:
            options2.append((0, 2))

        if tiles[2, 0] != NONE and tiles[2, 0] == tiles[0, 2] and tiles[1, 1] == NONE:
            options2.append((1, 1))

        if tiles[1, 1] != NONE and tiles[1, 1] == tiles[0, 2] and tiles[2, 0] == NONE:
            options2.append((2, 0))

        if len(options2) > 0:
            action = self.rnd.choice(options2)
        elif len(options) > 0:
            action = self.rnd.choice(options)
        else:
            action = self.rnd.choice(available)

        if not SILENT:
            print("{0} is making it's move: {1}, {2}".format(self.name, action[0] + 1, action[1] + 1))

        return action


class AIRandom(Player):
    def __init__(self, name='Random', exp_rate=0.2, learning_rate=0.1):
        Player.__init__(self, name)

    def make_move(self, tiles, available, player_id):
        if tiles[1, 1] == NONE:
            return 1, 1

        action = self.rnd.choice(available)

        if not SILENT:
            print("{0} is making it's move: {1}, {2}".format(self.name, action[0] + 1, action[1] + 1))

        return action


class Board:
    tiles: ndarray

    def __init__(self, p1, p2):
        self.tiles = None
        self.p1: Player = p1
        self.p2: Player = p2

        self.wins_p1 = 0
        self.wins_p2 = 0
        self.ties = 0
        self.games = 0

        self.reset()

        return

    def swap_players(self):
        p = self.p1
        self.p1 = self.p2
        self.p2 = p

        w = self.wins_p1
        self.wins_p1 = self.wins_p2
        self.wins_p2 = w

        return

    # board reset
    def reset(self):
        self.tiles = np.full((BOARD_ROWS, BOARD_COLS), NONE, int)
        self.p1.reset()
        self.p2.reset()

    # stats reset
    def reset_stats(self):
        self.games = 0
        self.wins_p1 = 0
        self.wins_p2 = 0
        self.ties = 0

    # get unique hash of current board state
    def get_hash(self):
        return str(self.tiles.reshape(BOARD_COLS * BOARD_ROWS))

    # get winner, PLAYER_1, PLAYER_2, TIE or NONE
    def get_winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.tiles[i, :]) == 3:
                return PLAYER_1
            if sum(self.tiles[i, :]) == -3:
                return PLAYER_2

        # col
        for i in range(BOARD_COLS):
            if sum(self.tiles[:, i]) == 3:
                return PLAYER_1
            if sum(self.tiles[:, i]) == -3:
                return PLAYER_2

        # diagonal
        diag_sum1 = sum([self.tiles[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.tiles[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])

        if diag_sum1 == 3 or diag_sum2 == 3:
            return PLAYER_1
        elif diag_sum1 == -3 or diag_sum2 == -3:
            return PLAYER_2

        # tie
        # no available positions
        if len(self.get_available_positions()) == 0:
            return TIE

        return NONE

    def get_available_positions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.tiles[i, j] == NONE:
                    positions.append((i, j))  # need to be tuple
        return positions

    def place_tile(self, player, position):
        self.tiles[position] = player

    # draw statistics
    def draw_stats(self):
        print("Wins: {0}: {1} ({2:.1%}), {3}: {4} ({5:.1%}), Ties: {6} ({7:.1%})".format(
            colored(self.p1.name, 'green' if self.wins_p1 > self.wins_p2 else 'red'),
            self.wins_p1,
            self.wins_p1 / self.games,
            colored(self.p2.name, 'green' if self.wins_p2 > self.wins_p1 else 'red'),
            self.wins_p2,
            self.wins_p2 / self.games,
            self.ties,
            self.ties /
            self.games)
        )
        return

    # draw board
    def draw(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            out = ''
            token = ''
            for j in range(0, BOARD_COLS):
                if self.tiles[i, j] == PLAYER_1:
                    token = 'X'
                if self.tiles[i, j] == PLAYER_2:
                    token = 'O'
                if self.tiles[i, j] == NONE:
                    token = '•'
                out += token + ' '
            print(out)

        print()

    def play(self):
        winner = NONE

        while winner == NONE:
            avail = self.get_available_positions()
            pos = self.p1.make_move(self.tiles, avail, PLAYER_1)

            if pos not in avail:
                print("Error!!!!")

            self.place_tile(PLAYER_1, pos)

            if not SILENT:
                self.draw()

            winner = self.get_winner()

            if winner == NONE:
                avail = self.get_available_positions()
                pos = self.p2.make_move(self.tiles, avail, PLAYER_2)
                if pos not in avail:
                    print("Error!!!!")

                self.place_tile(PLAYER_2, pos)

                if not SILENT:
                    self.draw()

                winner = self.get_winner()

        self.games += 1

        if winner == PLAYER_1:
            if not SILENT:
                print(self.p1.name + " won!")

            self.wins_p1 += 1
            self.p1.game_over(1)
            self.p2.game_over(0)
        elif winner == PLAYER_2:
            if not SILENT:
                print(self.p2.name + " won!")

            self.wins_p2 += 1
            self.p1.game_over(0)
            self.p2.game_over(1)
        else:
            if not SILENT:
                print("Tie")

            self.ties += 1
            self.p1.game_over(0.25)
            self.p2.game_over(0.5)

        return


def play_games_with_swap(games, p1, p2):
    board = Board(p1, p2)

    for game in range(games):
        board.play()
        board.reset()

        if game % 2 == 0 and game:
            board.swap_players()

        if game % 10000 == 0 and game > 0:
            board.draw_stats()
            board.reset_stats()

    if games % 2 == 0:
        board.swap_players()

    board.draw_stats()


def play_games(games, p1, p2):
    board = Board(p1, p2)

    for game in range(games * 2):
        board.play()
        board.reset()

        if game % 10000 == 0 and game > 0:
            board.draw_stats()
            board.reset_stats()

    board.draw_stats()

if __name__ == '__main__':
    start_time = perf_counter()

    play_games_with_swap(100000, AILearning("Learning 1", exp_rate=0.1, learning_rate=0.2), AIRandomEnhanced())
    #play_game(100000, AIRandomEnhanced(), AIRandom())

    #play_game(100, AIRandom(), AIRandom())

    if False:
        t1 = mp.Process(target=play_thread, args=(50000, AIRandomEnhanced("Enhanced"), AIRandom("Random")))
        t1.start()

        t2 = mp.Process(target=play_thread, args=(50000, AIRandomEnhanced("Enhanced"), AIRandomEnhanced("Enhanced 2")))
        t2.start()

        t3 = mp.Process(target=play_thread, args=(50000, AIRandomEnhanced("Enhanced"), AIRandom("Random")))
        t3.start()

        t4 = mp.Process(target=play_thread, args=(50000, AIRandomEnhanced("Enhanced"), AIRandomEnhanced("Enhanced 2")))
        t4.start()

        t5 = mp.Process(target=play_thread, args=(50000, AIRandom("Random"), AIRandom("Random 2")))
        t5.start()

        t6 = mp.Process(target=play_thread, args=(50000, AIRandom("Random"), AIRandom("Random 2")))
        t6.start()

        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        t6.join()

    end_time = perf_counter()

    print()
    print(f'It took{end_time- start_time: 0.1f} second(s) to complete.')
