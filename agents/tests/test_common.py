"""
test_common.py:
    Rigorous tests for each function in common.py
"""
# Import packages
import numpy as np
import agents.common as cc
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
    test_string = ('|--------------|\n'
                   '|              |\n'
                   '|              |\n'
                   '|              |\n'
                   '|              |\n'
                   '|      X       |\n' # PLAYER2 at (1, 3)
                   '|      O       |\n' # PLAYER1 at (0, 3)
                   '|--------------|\n'
                   '|0 1 2 3 4 5 6 |\n')

    # Function output
    out = cc.string_to_board(test_string)

    # Test output
    assert isinstance(out, np.ndarray)
    assert out.dtype == BoardPiece
    assert out.shape == (6, 7)
    assert out[0, 3] == PLAYER1
    assert out[1, 3] == PLAYER2


def test_apply_player_action():
    """
    test for apply_player_action()
        - PLAYER1 and PLAYER2 actions are played correctly
        - player actions can not go above full columns
    """
    # Initialize a test game
    board = cc.initialize_game_state()
    players = [PLAYER1, PLAYER2]
    player_turn = 0

    # Initialize test variables
    theoretical_sum = np.zeros((board.shape[1],))

    # Play with random agents until board is full
    while cc.possible_actions(board).shape[0] != 0:
        # Who is playing?
        player = players[player_turn]

        # Take a random action
        action = np.random.randint(board.shape[1], dtype=PlayerAction)

        # Function output -> apply action
        out = cc.apply_player_action(board=board, action=action, player=player, copy=True)

        # Test output
        # Calculate the sum of columns in the output
        empirical_sum = out.sum(axis=0)

        if np.isin(action, cc.possible_actions(board)):
            # The action can be taken and the corresponding player piece should affect the corresponding column
            # Update the theoretical sum of columns
            theoretical_sum[action] += player

            # Change turn for next round
            player_turn = (player_turn - 1) * -1

            # Update board
            board = out

            assert np.all(theoretical_sum == empirical_sum)
        else:
            # The action can not be taken because of a full column
            # The same player should play again and the board should have not been altered -> no theoretical update
            assert np.all(theoretical_sum == empirical_sum)

        # Test output
        assert isinstance(out, np.ndarray)
        assert out.dtype == BoardPiece
        assert out.shape == (6, 7)


def test_connected_four():
    """
    test for connected_four():
        - winning conditions are picked up
            - horizontal
            - vertical
            - diagonal 45deg
            - diagonal 135deg
        - win is possible for both players
        - no win is picked up also for both players
    """
    # Make transfer variables for ease of use
    n = NO_PLAYER
    o = PLAYER1
    x = PLAYER2

    for p, d in zip([PLAYER1, PLAYER2], [PLAYER2, PLAYER1]):
        # Test board for horizontal and vertical + distractions
        board1 = np.array([[p, p, p, p, n, n, n],
                           [d, n, d, n, d, n, d],
                           [n, n, n, n, n, n, n],
                           [n, d, n, d, n, d, n],
                           [n, n, n, n, n, n, n],
                           [d, n, d, n, d, n, d]])

        # Test board for 45deg diagonal and 135deg diagonal + distractions
        board2 = np.array([[p, d, n, d, n, d, n],
                           [n, p, n, n, n, n, n],
                           [d, n, p, n, d, n, d],
                           [n, n, n, p, n, n, n],
                           [n, d, n, d, n, d, n],
                           [n, n, n, n, n, n, n]])
        #
    assert cc.connected_four(test_board, cc.PLAYER1)


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
