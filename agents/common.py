"""
common.py:
    Common data types, classes, and functions used to play the connect 4 game
"""
# Import packages
import numpy as np
from scipy.signal import convolve2d
from enum import Enum
from typing import Optional
from typing import Callable, Tuple

# Define i/o data types
PlayerAction = np.int8
BoardPiece = np.int8

# Define Player pieces
NO_PLAYER = BoardPiece(0)
PLAYER1 = BoardPiece(1)
PLAYER2 = BoardPiece(2)

# Define game state class
class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

# Define save state class
class SavedState:
    # What is a saved state? todo
    pass

# Define moves for agents
GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

def initialize_game_state() -> np.ndarray:
    """
    Initializes connect 4 game board in a empty state (full of 0)
    :return:
        np.ndarray: zero matrix of shape (6, 7)
    """
    return np.zeros((6, 7), dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Converts a (6, 7) matrix filled with Player pieces into a user friendly printable string
    :param board:
        np.ndarray: (6, 7) matrix with Player pieces
    :return:
        str: user friendly printable connect four board of the current state of the game
    """
    # Define parsing dictionary
    p_dict = {' ': '',
             '0': '  ',
             '1': 'O ',
             '2': 'X ',
             '[': '|',
             ']': '|'}

    # Remove extra [...] and flip to make [0, 0] bottom corner
    str_board = f'{np.flipud(board)}'[1:-1]

    # Apply parsing dictionary
    for key in p_dict.keys():
        str_board = str_board.replace(key, p_dict[key])

    # Finish top and bottom borders + column numbers
    str_board = '|--------------|\n' + str_board + '\n|--------------|\n|0 1 2 3 4 5 6 |\n'
    return str_board


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Convert board string to (6, 7) matrix filled with Player pieces
    :param pp_board:
        str: user friendly printable connect four board
    :return:
        np.ndarray: (6, 7) matrix with Player pieces
    """
    # Define parsing dictionary
    p_dict = {'  ': '0',
              'O ': '1',
              'X ': '2',
              '|': ''}

    # Remove top and bottom edges as well as columns
    board = pp_board[17:-35]

    # Apply parsing dictionary
    for key in p_dict.keys():
        board = board.replace(key, p_dict[key])

    # Split into rows
    board = board.split('\n')

    # Change back to np.ndarray type
    board = np.stack([np.fromiter(b, dtype=BoardPiece) for b in board])

    # Flip vertically to have (0, 0) on top left corner
    board = np.flipud(board)
    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) -> np.ndarray:
    """
    Given a players column choice, put the specifics Player piece in the board. The piece will always go on the top, of
    the selected column. If an action is not possible, no action is taken and the board's state is not altered.
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param action:
        PLayerAction: board column in which a Player piece will be placed on top
    :param player:
        BoardPiece: type of Player piece to be placed on the boards. Who is playing?
    :param copy:
        bool: if True, a copy of the board is made...
    :return:
        np.ndarray: game board with the new player action added to its state
    """
    # What is up with the copy part? todo

    # Check if the action is executable
    if np.isin(action, possible_actions(board)):  # no full column
        # Find location row of the action
        i = np.argwhere(board[:, action] == NO_PLAYER).min()

        # Update the board
        board[i, action] = player
        return board
    else:
        # No action taken
        return board


def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, ) -> bool:
    """
    Check in the current game board if the specified player has won the game or not. This is implemented using
    convolutions
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: type of Player piece to check for winning condition. Did player win?
    :param last_action:
        PlayerAction: last action taken on the board
    :return:
        bool: flag indicating if the player has won or not
    """
    # Define kernels for winning conditions
    horizontal_4 = np.array([[1, 1, 1, 1]])
    vertical_4 = np.array([[1], [1], [1], [1]])
    diag_4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    diag_inv_4 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    win_conditions = [horizontal_4, vertical_4, diag_4, diag_inv_4]

    # Make a board that contains only the specified player pieces
    player_board = initialize_game_state()
    player_board[board == player] = player

    # Convolve kernels on the game board to check for a win
    win = np.any([(convolve2d(player_board, k, mode='valid') == 4).any() for k in win_conditions])
    return win


def check_end_state(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, ) -> GameState:
    """
    Checks the game board and checks for the current state of the game for a specific player
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: type of Player piece to check for current game state. What is the player doing?
    :param last_action:
        PlayerAction: last action taken on the board
    :return:
        GameState: current state of the game for a specific player
    """
    # Check for a win condition
    if connected_four(board, player):
        return GameState.IS_WIN
    elif possible_actions(board).shape[0] == 0:   # check for no more possible actions... a draw
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING

def possible_actions(board: np.ndarray) -> np.ndarray:
    """
    Checks the game board columns to see which ones are not full and returns an array of possible columns in which
    it is still possible to play a Player piece
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :return:
        np.ndarray: vector with possible player actions to take
    """
    return np.argwhere(np.sum(board == NO_PLAYER, axis=0) > 0)
