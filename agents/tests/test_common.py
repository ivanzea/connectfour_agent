"""
test_common.py:
    Rigorous tests for each function in common.py
"""
# Import packages
import numpy as np
import agents.common as cc
from agents.common import PlayerAction, BoardPiece
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
                   '|      X       |\n'  # PLAYER2 at (1, 3)
                   '|      O       |\n'  # PLAYER1 at (0, 3)
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
        - piece is always put on top of the column

        Implementation:
        > Keep track of the where the player actions are taken (which column) and add the Player piece
          value to an array of column sums. This way, you know exactly what sum should be in theory at
          every column every move. Compare it against an empirical value of the column sum of the functions
          board output. If they are the same, the correct player action is taken in the appropriate column.
          All moves are taken completely at random until the board is full.
            If the columns are full, the same player should not be able to play a piece, this means, that
          the column sum is not altered, and that the same player will put a piece next turn. The sums
          between the empirical and theoretical columns can be used to check this too.
            The previous is true even in a weird case in which the numbers are put in the columns but not
          on the top of it. We also have to check the consecutive zeros at bottom (using helper function)
    """

    # Internal helper function
    def zero_runs(board: np.ndarray) -> np.ndarray:
        """
        Identify blocks of continues zeros and return initial and end index for each block
        :param board:
            np.ndarray: current state of the board, filled with Player pieces
        :return:
            np.ndarray: indexes of blocks of consecutive zeros
        """
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(board, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))

        # Runs start and end where absdiff is 1.
        return np.where(absdiff == 1)[0].reshape(-1, 2)

    # Initialize a test game
    board = cc.initialize_game_state()
    players = [PLAYER1, PLAYER2]
    player_turn = 0

    # Initialize test variables
    theoretical_sum = np.zeros((board.shape[1],))

    # Play with random agents until board is full
    while (np.arange(board.shape[1])[board[-1, :] ==  NO_PLAYER]).shape[0] != 0:
        # Who is playing?
        player = players[player_turn]

        # Take a random action
        action = np.random.randint(board.shape[1], dtype=PlayerAction)

        # Function output -> apply action
        out = cc.apply_player_action(board=board, action=action, player=player, copy=True)

        # Test output
        # Calculate the sum of columns in the output
        empirical_sum = out.sum(axis=0)

        if np.isin(action, (np.arange(board.shape[1])[board[-1, :] ==  NO_PLAYER])):
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

        # Check that the pieces are put always on the columns bottom
        assert np.all([j[0, 1] == 6 if j.shape[0] else True for j in [zero_runs(i) for i in board.T]])

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
            - diagonal l
            - diagonal r
        - win is possible for both players
        - no win is picked up also for both players

    Implementation:
    > Make a test board with a 4 in a row horizontal pattern and the rest filled with the opponents
      pieces and zeros (noise). Shift the board column to the right and use it to asses for a
      winning condition. Repeat this until the board has been shifted over all positions and also
      perform this over the rows to cover all possible positions where the 4 in row horizontal win
      condition can appear. On the process, the pattern will also be broken when the matrix wraps
      around itself, making 3 in a row and 2 in a row, this can also be checked to see that there
      should not be winning conditions in the board. The previous steps can be done for the vertical,
      diagonal R, and diagonal L, to test all possible winning conditions. This will basically be
      a full permutation test of winning conditions in the board plus added features like checking
      for no winning conditions. We also should repeat this process for winning boards for player 1
      and for player 2.
    """
    # Make transfer variables for ease of use
    n = NO_PLAYER
    o = PLAYER1
    x = PLAYER2

    # Loop between playesrs
    for p, d in zip([PLAYER1, PLAYER2], [PLAYER2, PLAYER1]):
        # Test board for horizontal and vertical + distractions
        board1 = np.array([[p, p, p, p, n, n, n],
                           [d, n, d, n, d, n, d],
                           [n, n, n, n, n, n, n],
                           [n, d, n, d, n, d, n],
                           [n, n, n, n, n, n, n],
                           [d, n, d, n, d, n, d]])

        # Test board for right diagonal and left diagonal + distractions
        board2 = np.array([[p, d, n, d, n, d, n],
                           [n, p, n, n, n, n, n],
                           [d, n, p, n, d, n, d],
                           [n, n, n, p, n, n, n],
                           [n, d, n, d, n, d, n],
                           [n, n, n, n, n, n, n]])

        # This will perform an extensive permutation testing
        for (i, j), _ in np.ndenumerate(board1):
            # Horizontal
            h = cc.connected_four(board=np.roll(np.roll(board1, i, axis=1), j, axis=0), player=p)

            # Vertical
            v = cc.connected_four(board=np.roll(np.roll(board1, i, axis=1), j, axis=0).T, player=p)

            # Diagonal L
            dl = cc.connected_four(board=np.roll(np.roll(board2, i, axis=1), j, axis=0), player=p)

            # Diagonal R
            dr = cc.connected_four(board=np.fliplr(np.roll(np.roll(board2, i, axis=1), j, axis=0)), player=p)

        # Winning condition met
        if (i < board1.shape[0] - 4) & (j < board1.shape[1] - 4):
            assert h
            assert v
            assert dl
            assert dr
        else:
            assert ~h
            assert ~v
            assert ~dl
            assert ~dr

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
