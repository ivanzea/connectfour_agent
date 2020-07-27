"""
test_mcts.py:
    Check that an agent takes practical moves... 'correct ones'
"""
# Import packages
import numpy as np
import agents.common as cc
from agents.common import PlayerAction, SavedState
from agents.common import PLAYER1, PLAYER2
from agents.agent_mcts.mcts_optim import  Node
from agents.agent_mcts.mcts_optim import mcts
from agents.agent_mcts.mcts_optim import generate_move_mcts as generate_move

def test_generate_move_mcts():
    """
    Must return a player action, and a SavedState in a tuple...
    The SavedState contains a root + node of class Node
    """
    for ply in [PLAYER1, PLAYER2]:
        out = generate_move(board=cc.initialize_game_state(), player=ply, saved_state=None)
        assert type(out) == tuple
        assert isinstance(out[0], PlayerAction)
        assert isinstance(out[1], SavedState)
        assert isinstance(out[1].root, Node)
        assert isinstance(out[1].node, Node)

def test_mcts():
    """
    Must return a tuple and position 0 must be a player action
    Check this with different possible parameters choosing random values
    """
    # Parameters
    rep = 100
    ply = [PLAYER1, PLAYER2]
    ss = None  # maximum depth

    for i in range(rep):
        action, root, node = mcts(board=cc.initialize_game_state(), player=np.random.choice(ply),
                                saved_state=ss, c=np.random.rand(), max_t=0.5)

        assert isinstance(action, PlayerAction)
        assert isinstance(root, Node)
        assert isinstance(node, Node)

        ss = SavedState(root=root, node=node)

def test_node_class_initialization():
    # root node initialization
    root = Node(state=cc.initialize_game_state(), player=PLAYER1)

    assert (root.n == 1)
    assert (root.r == 0)
    assert np.all(root.s == cc.initialize_game_state())
    assert (root.player == PLAYER1)
    assert (root.next_player == PLAYER2)
    assert (root.status == cc.GameState.STILL_PLAYING)
    assert (root.children == {})
    assert (root.parent is None)

def test_node_class_child_expansion():
    """
    Given that the connect 4 game has 7 possible actions, every 7 actions there should be an increase in
    depth when using a c value of 0. Also, the number of expanded nodes should be increasing fully before
    a new depth is reached.

    This test that the function of next_node works appropriately for both cases, the expansion, and the
    selection of most urgent child... just remember that one child must have a bigger r/n value... so set
    the first in each depth as a bigger one.
    """
    # root node initialization
    root = Node(state=cc.initialize_game_state(), player=PLAYER1)
    players = [PLAYER1, PLAYER2]

    # Run until an end node is reached... that is when this expansion logic stops working
    _node = root
    n_iter = 0
    while _node.status == cc.GameState.STILL_PLAYING:
        # Make a new node
        _node = root.next_node(c=0.0)
        assert isinstance(_node, Node)

        # End node reached
        if _node.status != cc.GameState.STILL_PLAYING:
            break

        # Check for the number of expansions
        n_exp = n_iter % 7 + 1  # theoretical number of expansion in the new node
        assert (n_exp == len(list(_node.parent.children.keys())))

        # Theoretical depth
        n_depth = n_iter // 7 + 1

        # Player should be changing at each new depth
        if bool(n_depth % 2):
            assert (_node.player == players[1])
        else:
            assert (_node.player == players[0])

        # Calculate current depth going up to the root node
        depth = 0
        while None != _node.parent:
            depth += 1
            _node = _node.parent

        assert (n_depth == depth)

        # Increase the iterations
        n_iter += 1


def test_node_class_simulation():
    # In the following board, no player has won, but any move by PLAYER1 results in a win
    p1_board = ('|--------------|\n'
                '|      O O O   |\n'
                '|O O O X X O X |\n'
                '|O X X O X X X |\n'
                '|O O X X O O O |\n'
                '|X O O X O X O |\n'
                '|X O O X O O O |\n'
                '|--------------|\n'
                '|0 1 2 3 4 5 6 |\n')
    p1_board = cc.string_to_board(p1_board)

    # Do the same for PLAYER 2
    p2_board = p1_board.copy()
    p2_board[p2_board == PLAYER1] = 3
    p2_board[p2_board == PLAYER2] = PLAYER1
    p2_board[p2_board == 3] = PLAYER2

    draw_board = ('|--------------|\n'
                 '|  X   O X     |\n'
                 '|O O O X O O O |\n'
                 '|X X X O X X X |\n'
                 '|O O X X X O O |\n'
                 '|X O O X O X O |\n'
                 '|X O O X O O O |\n'
                 '|--------------|\n'
                 '|0 1 2 3 4 5 6 |\n')
    draw_board = cc.string_to_board(draw_board)

    # P1 should win r=1
    p1_node = Node(state=p1_board, player=PLAYER1)
    assert (1 == p1_node.simulation())

    # P2 should lose r=-1
    p1_node = Node(state=p1_board, player=PLAYER2)
    assert (-1 == p1_node.simulation())

    # P2 should win r=1
    p2_node = Node(state=p2_board, player=PLAYER2)
    assert (1 == p2_node.simulation())

    # P1 should lose r=-1
    p2_node = Node(state=p2_board, player=PLAYER1)
    assert (-1 == p2_node.simulation())

    # Welp... it should be a draw no matter who plays
    draw_node = Node(state=draw_board, player=PLAYER1)
    assert (0 == draw_node.simulation())

    draw_node = Node(state=draw_board, player=PLAYER2)
    assert (0 == draw_node.simulation())


def test_node_class_backprop():
    # Make a tree structure of variable depth... repeat n times
    for n in range(100):
        # Make tree
        root = Node(state=cc.initialize_game_state(), player=PLAYER1)

        _node = root
        total_reward = 0
        while _node.status == cc.GameState.STILL_PLAYING:
            _node = root.next_node(c=0.0)

        # Backpropagate a random reward several times and keep track of theoretical root value
        for i in range(1000):
            rand_reward = np.random.randint(low=-10, high=10)
            total_reward += -rand_reward if _node.player == root.player else rand_reward
            _node.backprop(delta=rand_reward)

        # Check for the same values as theoretical
        assert (total_reward == root.r)
        assert (1001 == root.n)

        # Make sure the end node rules are also working... or at leas one of them Rule 1
        if _node.r == np.inf:
            assert (_node.parent.r == -np.inf)

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
                agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER1, None, max_t=0.5)
                assert (expected_block_column + shift == agent_action)

                # Play to block
                agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER2, None, max_t=0.5)
                assert(expected_block_column+shift == agent_action)

                # Change player, test with opposite player
                temp = test_board.copy()
                test_board[temp == PLAYER1] = PLAYER2
                test_board[temp == PLAYER2] = PLAYER1

                # Play to block
                agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER1, None, max_t=0.5)
                assert (expected_block_column + shift == agent_action)

                # Play to win
                agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER2, None, max_t=0.5)
                assert (expected_block_column + shift == agent_action)

        expected_block_column = vert_block
        test_board = test_vert

        # Play for the win
        agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER1, None, max_t=0.5)
        assert (expected_block_column + shift == agent_action)

        # Play to block
        agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER2, None, max_t=0.5)
        assert (expected_block_column + shift == agent_action)

        # Change player, test with opposite player
        temp = test_board.copy()
        test_board[temp == PLAYER1] = PLAYER2
        test_board[temp == PLAYER2] = PLAYER1

        # Play to block
        agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER1, None, max_t=0.5)
        assert (expected_block_column + shift == agent_action)

        # Play to win
        agent_action, _ = generate_move(np.roll(test_board, shift, axis=1), PLAYER2, None, max_t=0.5)
        assert (expected_block_column + shift == agent_action)
