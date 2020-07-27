"""
test_minimax.py:
    Check that an agent takes practical moves... 'correct ones'
"""
# Import packages
import numpy as np
import agents.common as cc
from agents.common import PlayerAction, BoardPiece
from agents.common import NO_PLAYER, PLAYER1, PLAYER2
from agents.agent_minimax import minimax
from agents.agent_minimax import generate_move


def test_generate_move_minimax():
    """
    Must return a player action
    """
    for ply in [PLAYER1, PLAYER2]:
        out = generate_move(board=cc.initialize_game_state(), player=ply, saved_state=None)
        assert type(out) == tuple
        assert isinstance(out[0], PlayerAction)


def test_minimax():
    """
    Must return a tuple and position 0 must be a player action
    Check this with different possible parameters choosing random values
    """
    # Parameters
    rep = 10  # careful it was actually 100 when tested but takes a loooooong time
    ply = [PLAYER1, PLAYER2]
    max_depth = 8  # maximum depth

    for i in range(rep):
        out = minimax.minimax(board=cc.initialize_game_state(), player=np.random.choice(ply),
                              score_dict=np.random.randint(low=1, high=100, size=(5,)),
                              depth=np.random.randint(low=1, high=max_depth+1),
                              alpha=np.random.randint(low=-1000, high=1000), beta=np.random.randint(low=-1000, high=1000),
                              maxplayer=np.random.choice(ply))

        assert type(out) == tuple
        assert isinstance(out[0], PlayerAction)


def test_board_score():
    """
    5 patterns that give points (score) given a score_dict (score dictionary):

        [p, p, p, n]
        [p, p, n, p]
        [p, p, n, n]
        [p, n, p, n]
        [p, n, n, p]

    The way to check the scoring is done well... is to manually check so... check that the score are the same
    """
    # Parameters
    n = NO_PLAYER
    score_dict = np.array([5, 4, 3, 2, 1])
    test_board = ('|--------------|\n'
                  '|              |\n'
                  '|              |\n'
                  '|              |\n'
                  '|  X X   O     |\n'
                  '|  X X O O   O |\n'
                  '|O O O X O X X |\n'
                  '|--------------|\n'
                  '|0 1 2 3 4 5 6 |\n')
    test_board = cc.string_to_board(test_board)
    manual_score = [20, 14]  # P1 and P2 scores

    assert(manual_score[0] == minimax.board_score(board=test_board, player=PLAYER1, score_dict=score_dict))
    assert(manual_score[1] == minimax.board_score(board=test_board, player=PLAYER2, score_dict=score_dict))


def test_easy_block_win():
    """
    Test that the agent blocks an opponent from winning and take winning moves

    Remember:
        'O ': PLAYER1
        'X ': PLAYER2
        '  ': NO_PLAYER
    """
    # Test horizontal block
    test_horz = ('|--------------|\n'
                 '|              |\n'
                 '|              |\n'
                 '|              |\n'
                 '|              |\n'
                 '|              |\n'
                 '|O O O       X |\n'
                 '|--------------|\n'
                 '|0 1 2 3 4 5 6 |\n')
    test_horz = cc.string_to_board(test_horz)
    horz_block = 3

    # Test horizontal block
    test_dia1 = ('|--------------|\n'
                 '|              |\n'
                 '|              |\n'
                 '|              |\n'
                 '|    O O       |\n'
                 '|  O X X       |\n'
                 '|O X X O       |\n'
                 '|--------------|\n'
                 '|0 1 2 3 4 5 6 |\n')
    test_dia1 = cc.string_to_board(test_dia1)
    dia1_block = 3

    # Test horizontal block
    test_dia2 = ('|--------------|\n'
                 '|              |\n'
                 '|              |\n'
                 '|              |\n'
                 '|O O           |\n'
                 '|X X O         |\n'
                 '|O X X O       |\n'
                 '|--------------|\n'
                 '|0 1 2 3 4 5 6 |\n')
    test_dia2 = cc.string_to_board(test_dia2)
    dia2_block = 0

    # Test horizontal block
    test_vert = ('|--------------|\n'
                 '|              |\n'
                 '|              |\n'
                 '|              |\n'
                 '|O             |\n'
                 '|O             |\n'
                 '|O             |\n'
                 '|--------------|\n'
                 '|0 1 2 3 4 5 6 |\n')
    test_vert = cc.string_to_board(test_vert)
    vert_block = 0

    # We asses if the agent plays at the target column and we will move this position along the horizontal axis. Then we
    # change make the same test but changing the player pieces to check the agent can play with both pieces.
    for shift in range(7):
        # horz, dia1 and dia2 can only be shifted 3 times
        if shift <= 3:
            for expected_block_column, test_board in zip([horz_block, dia1_block, dia2_block], [test_horz, test_dia1, test_dia2]):
                # Play for the win
                agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER1, None)
                assert (expected_block_column + shift == agent_action)

                # Play to block
                agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER2, None)
                assert(expected_block_column+shift == agent_action)

                # Change player, test with opposite player
                temp = test_board.copy()
                test_board[temp == PLAYER1] = PLAYER2
                test_board[temp == PLAYER2] = PLAYER1

                # Play to block
                agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER1, None)
                assert (expected_block_column + shift == agent_action)

                # Play to win
                agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER2, None)
                assert (expected_block_column + shift == agent_action)

        expected_block_colmn = vert_block
        test_board = test_vert

        # Play for the win
        agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER1, None)
        assert (expected_block_column + shift == agent_action)

        # Play to block
        agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER2, None)
        assert (expected_block_column + shift == agent_action)

        # Change player, test with opposite player
        temp = test_board.copy()
        test_board[temp == PLAYER1] = PLAYER2
        test_board[temp == PLAYER2] = PLAYER1

        # Play to block
        agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER1, None)
        assert (expected_block_column + shift == agent_action)

        # Play to win
        agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER2, None)
        assert (expected_block_column + shift == agent_action)