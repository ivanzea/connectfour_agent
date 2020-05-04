from typing import List
import numpy as np
import agents.common as cc


def test_initialize_game_state():
    ret = cc.initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    assert np.all(ret == 0)


def test_pretty_print_board():
    board = cc.initialize_game_state()
    ret = cc.pretty_print_board(board)

    assert isinstance(ret, str)


def test_string_to_board():
    board = cc.initialize_game_state()
    pp_board = cc.pretty_print_board(board)
    ret = cc.string_to_board(pp_board)

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)


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
