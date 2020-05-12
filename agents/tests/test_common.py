"""
test_common.py:
    Rigorous tests for each function in common.py
"""
# Import packages
import numpy as np
import agents.common as cc
from itertools import combinations
from agents.common import  PlayerAction, BoardPiece
from agents.common import NO_PLAYER, PLAYER1, PLAYER2

def test_initialize_game_state():
    """
    test for initialize_game_state()
        - np.ndarray type
        - dimensions of (6, 7)
        - filled with NO_PLAYER pieces -> also being BoardPiece
    """
    # Function output
    out = cc.initialize_game_state()

    # Test output
    assert isinstance(out, np.ndarray)
    assert out.dtype == BoardPiece
    assert out.shape == (6, 7)
    assert np.all(out == NO_PLAYER)


def test_pretty_print_board():
    """
    test for pretty_print_board
        - be str type
        - have a length of 153
        - PLAYER1 and PLAYER2 tokens placed correctly [ visual inspection ]
    """
    # Make transfer variables for ease of use
    n = NO_PLAYER
    o = PLAYER1
    x = PLAYER2

    # Define test board:
    #   ARROW pointing UP
    #   PLAYER1 -> 'O' make the head of the arrow
    #   PLAYER2 -> 'X' make the body of the arrow
    test_board = np.array([[n, n, n, o, n, n, n],
                           [n, n, o ,o, o, n, n],
                           [n, o, o, o, o, o, n],
                           [n, n, x, x, x, n, n],
                           [n, n, x, x, x, n, n],
                           [n, n, x, x, x, n, n],])

    # Function output
    out = cc.pretty_print_board(test_board)

    # Test output
    assert isinstance(out, str)
    assert len(out) == 153

    # Visual inspection
    print('\n\nThis should be an ARROW pointing DOWN \n \'O\': head\n \'X\': body')
    print(out)


def test_string_to_board():
    """
    test for string_to_board():
        - np.ndarray type
        - dimensions of (6, 7)
        - BoardPiece type
        - Correct conversion of ['O', 'X'] -> [PLAYER1, PLAYER2] types

    """
    test_string = ('|--------------|\n',
                   '|              |\n',
                   '|              |\n',
                   '|              |\n',
                   '|              |\n',
                   '|      X       |\n', # PLAYER2 at (1, )
                   '|      O       |\n',
                   '|--------------|\n',
                   '|0 1 2 3 4 5 6 |\n')


    # Test output
    assert isinstance(out, np.ndarray)
    assert out.dtype == BoardPiece
    assert out.shape == (6, 7)
    assert np.all(out == NO_PLAYER)


def test_apply_player_action():
    board = cc.initialize_game_state()
    player_turns = np.array([1, 2, 1, 2, 1, 2, 1], dtype=np.int8)

    for i, player in enumerate(player_turns):
        cp_board = board.copy()
        board = cc.apply_player_action(board, np.int8(2), player)

        if i < player_turns.shape[0]-1:
            assert (cp_board != board).all
        else:
            assert (cp_board == board).all  # no action should be taken

        print(cc.pretty_print_board(board))


def test_connected_four():
    # Horizontal
    test_board = np.array([[0, 0, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert cc.connected_four(test_board, cc.PLAYER1)

    # Vertical
    test_board = np.array([[0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 1, 1, 0, 0, 0],
                           [0, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert cc.connected_four(test_board, cc.PLAYER1)

    # Diagonal I
    test_board = np.array([[0, 1, 1, 0, 0, 0, 0],
                           [1, 0, 1, 1, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert cc.connected_four(test_board, cc.PLAYER1)

    # Diagonal II
    test_board = np.array([[0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 1, 1, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0],
                           [1, 0, 1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert cc.connected_four(test_board, cc.PLAYER1)

    # No connection
    test_board = np.array([[0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 1, 1, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert not cc.connected_four(test_board, cc.PLAYER1)

    # player 2 - no connection
    test_board = np.array([[0, 2, 1, 2, 2, 1, 2],
                           [1, 1, 1, 1, 2, 2, 0],
                           [0, 1, 2, 1, 0, 0, 0],
                           [2, 0, 1, 0, 0, 0, 0],
                           [0, 0, 2, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert not cc.connected_four(test_board, cc.PLAYER2)

    # Diagonal II player 2
    test_board = np.array([[2, 2, 1, 2, 2, 1, 0],
                           [1, 2, 1, 1, 1, 2, 0],
                           [1, 1, 2, 1, 2, 0, 0],
                           [1, 2, 1, 2, 0, 0, 0],
                           [2, 1, 2, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert cc.connected_four(test_board, cc.PLAYER2)

def test_end_state():
    # Player one should still be playing
    test_board = np.array([[2, 2, 1, 2, 2, 1, 1],
                           [1, 2, 1, 1, 1, 2, 1],
                           [1, 1, 2, 2, 2, 1, 1],
                           [1, 2, 1, 1, 1, 2, 2],
                           [2, 1, 2, 2, 1, 1, 1],
                           [2, 2, 2, 1, 2, 0, 2]], dtype=np.int8)
    assert cc.check_end_state(test_board, cc.PLAYER1) == cc.GameState(0)

    # Player two should have a win condition
    test_board = np.array([[2, 2, 1, 2, 2, 1, 1],
                           [1, 2, 1, 1, 1, 2, 1],
                           [1, 1, 2, 2, 2, 1, 1],
                           [1, 2, 1, 1, 1, 2, 2],
                           [2, 1, 2, 2, 1, 1, 1],
                           [2, 2, 2, 2, 2, 0, 2]], dtype=np.int8)
    assert cc.check_end_state(test_board, cc.PLAYER2) == cc.GameState(1)

    # Tie for P1
    test_board = np.array([[2, 2, 1, 2, 2, 1, 1],
                           [1, 2, 1, 1, 1, 2, 1],
                           [1, 1, 2, 2, 2, 1, 1],
                           [1, 2, 1, 1, 1, 2, 2],
                           [2, 1, 2, 2, 1, 1, 1],
                           [2, 2, 2, 1, 2, 1, 2]], dtype=np.int8)
    assert cc.check_end_state(test_board, cc.PLAYER1) == cc.GameState(-1)

    # Tie for P2
    test_board = np.array([[2, 2, 1, 2, 2, 1, 1],
                           [1, 2, 1, 1, 1, 2, 1],
                           [1, 1, 2, 2, 2, 1, 1],
                           [1, 2, 1, 1, 1, 2, 2],
                           [2, 1, 2, 2, 1, 1, 1],
                           [2, 2, 2, 1, 2, 1, 2]], dtype=np.int8)
    assert cc.check_end_state(test_board, cc.PLAYER2) == cc.GameState(-1)
