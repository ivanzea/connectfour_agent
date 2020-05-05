import numpy as np
import re
from enum import Enum
from itertools import groupby
from typing import Optional
from typing import Callable, Tuple


PlayerAction = np.int8
BoardPiece = np.int8
NO_PLAYER = BoardPiece(0)
PLAYER1 = BoardPiece(1)
PLAYER2 = BoardPiece(2)


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    return np.zeros((6, 7), dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    str_board = f'{np.flipud(board)}'[1:-1]  # remove extra [...] ends flip to make [0, 0] bottom corner
    str_board = re.sub(r'(\[|\])', '|', str_board)  # add lines at end
    str_board = str_board.replace('\n ', '\n')  # align columns
    str_board = re.sub(r'0 ?', '  ', str_board)  # if there is a 0
    str_board = re.sub(r'1 ?', 'O ', str_board)  # if there is a 1
    str_board = re.sub(r'2 ?', 'X ', str_board)  # if there is a 2
    str_board += '\n|--------------|\n|0 1 2 3 4 5 6 |'  # add column index

    return str_board


def string_to_board(pp_board: str) -> np.ndarray:
    board = re.sub(r'(?s)\n\|-+.+', '', pp_board)  # get rid of column index
    board = board.replace('  ', '0')  # put 0 back
    board = board.replace('O ', '1')  # put 1 back
    board = board.replace('X ', '2')  # put 2 back
    board = board.replace('|', '')  # remove borders
    board = board.split('\n')  # split into rows
    board = np.flipud(np.stack([np.fromiter(b, dtype=BoardPiece) for b in board]))  # change back to ndarray

    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) -> np.ndarray:
    # Find possible actions
    poss_actions = np.argwhere(np.sum(board == NO_PLAYER, axis=0) > 0)

    # What is up with the copy part? todo

    # Check if the action is executable
    if np.isin(action, poss_actions):  # no full column
        # Find location row of the action
        i = np.argwhere(board[:, action] == NO_PLAYER).min()

        # Update the board
        board[i, action] = player

        return board
    else:
        # No action taken
        return board


def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, ) -> bool:
    player_board = initialize_game_state()
    player_board[board == player] = PLAYER1

    h_win = [np.array(i) for i in player_board]
    v_win = [np.array(i) for i in player_board.T]
    diagI_win = [player_board.diagonal(i) for i in np.arange(-(player_board.shape[0] - 1), (player_board.shape[1]))]
    diagII_win = [np.rot90(player_board).diagonal(i) for i in
                  np.arange(-(player_board.T.shape[0] - 1), (player_board.T.shape[1]))]

    possible_wins = [h_win, v_win, diagI_win, diagII_win]
    return np.array([(np.convolve(line, [1, 1, 1, 1]) == 4).any() for i in possible_wins for line in i]).any()


def check_end_state(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, ) -> GameState:
    # Find possible actions
    poss_actions = np.argwhere(np.sum(board == NO_PLAYER, axis=0) > 0)

    # Check for a win condition
    if connected_four(board, player):
        return GameState.IS_WIN
    elif poss_actions.shape[0] == 0:   # check for no more possible actions... a draw
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]
