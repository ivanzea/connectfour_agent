"""
mcts_base.py
    Implementation of a connect 4 agent that plays using the monte carlo tree search algorithm
"""
# Import packages
import time

from typing import Tuple, Optional

import numpy as np

import agents.common as cc
from agents.common import BoardPiece, SavedState, PlayerAction
from agents.common import PLAYER1, PLAYER2, NO_PLAYER


def generate_move_mcts(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], c=1.25, max_t=4.99,
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose an action based on the mcts algorithm
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: player piece to check for best move
    :param c:
        float: exploration coefficient
    :param max_t:
        float: maximum time to perform mcts
    :param saved_state:
        SavedState: standard generate_move input... not used in this case
    :return:
        PlayerAction: column to play based on the mcts algorithm
    """
    # Apply mcts algorithm
    action, root, saved_state = mcts(board=board, player=player, saved_state=saved_state, c=c, max_t=max_t)

    return action, SavedState(root=root, node=saved_state)


def mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], c: float, max_t: float):
    # Create root node
    root = []
    if None != saved_state:  # in case a tree was saved
        for node in list(saved_state.node.children.values()):
            if np.all(node.s == board):
                root = node
        if not root:  # last move is not part of the saved tree
            player = PLAYER2 if player == PLAYER1 else PLAYER1  # inverse player initialization given the description of player for class node
            root = Node(board, player)
    else:
        player = PLAYER2 if player == PLAYER1 else PLAYER1  # inverse player initialization given the description of player for class node
        root = Node(board, player)

    # Search tree while there is still time
    start = time.time()
    while time.time() - start <= max_t:
        # Choose next node to visit
        new_node = root.next_node(c=c)

        # Simulate a game till end state is reached
        delta = new_node.simulation()

        # Back propagate
        new_node.backprop(delta=delta)

    # When time runs out, pick the best option
    best_child, action = ucb1_func(node=root, c=0)
    best_child.parent = None
    return action, root, best_child

class Node:
    """
    class Node
    This class contains the information about the state and how it has been used during the tree search
    """
    # Initialize node
    def __init__(self, state: np.ndarray, player: BoardPiece, parent=None):
        """
        Node initialization
        :param state:
            np.ndarray: game state that the node represents
        :param player:
            BoardPiece: player that acted last and lead to the current state
        :param parent:
            node: parent node
        """
        self.n = 1
        self.r = 0.0
        self.s = state
        self.children = {}
        self.parent = parent
        self.player = player
        self.next_player = PLAYER2 if player == PLAYER1 else PLAYER1
        self.status = cc.check_end_state(self.s, self.player)

        # Who is the root player
        if None == self.parent:
            self.root_player = self.next_player
        else:
            self.root_player = self.parent.root_player

        # End state values
        if self.status == cc.GameState.IS_WIN:
            self.r = np.inf


    def next_node(self, c: float):
        # Define the current node
        _node = self

        # Repeat until terminal node is found
        while _node.status == cc.GameState.STILL_PLAYING:
            # Apply tree search policy
            actions = np.delete(poss_actions(_node.s), list(_node.children.keys()))  # find possible unexplored actions v0.0
            #actions = [act for act in poss_actions(_node.s) if not act in list(_node.children.keys())]  # find possible unexplored actions v1.0

            # Check for unexplored children nodes
            if len(actions) != 0:
                # Take a random actions
                a = np.random.choice(actions)  # select action

                # Add new child node
                _node.children[a] = Node(state=cc.apply_player_action(_node.s, a, _node.next_player, copy=True),
                                         player=_node.next_player, parent=_node)

                return _node.children[a]
            else:  # explore which is the best child using UBC1
                _node, _ = ucb1_func(_node, c)

        return _node

    def simulation(self):
        # Take random actions until an end state is reached
        player = self.player
        opponent = self.next_player

        # Perform simulation until end state is reached
        s = self.s.copy()
        players = [player, opponent]
        while cc.check_end_state(s, players[0]) == cc.GameState.STILL_PLAYING:
            # Take a random action
            s = cc.apply_player_action(s, np.random.choice(poss_actions(s)), players[1])

            # Change player
            players = players[::-1]

        # Check who won the match
        if cc.check_end_state(s, players[0]) == cc.GameState.IS_WIN:
            return 1 if players[0] == player else -1  # if the winner is equal to the initial player, r = 1
        else:
            return 0.0  # case of a tie

    def backprop(self, delta):
        _node = self

        # Check for terminal node and use special case back propagation
        if _node.r == np.inf:
            # Inf value backprop
            inf_rule = True  # check thant propagation rules still apply

            while inf_rule and (None != _node.parent):
                # Rule 1 -> I wont let my opponent take a win move!
                if _node.r == np.inf:
                    _node.parent.r = -np.inf
                    _node = _node.parent
                # Rule 2 -> If all options are bad for my opponent... I will obviously make that move <3
                elif all([n.r == -np.inf for n in _node.parent.children.values()]):
                    _node.parent.r = np.inf
                    _node = _node.parent
                # No rule applied... so lets move on...
                else:
                    inf_rule = False
        # Normal back propagation
        # Back propagate until the root node is reached
        while None != _node:
            # Update n and r
            _node.n += 1
            _node.r += delta

            delta = delta * -1  # P1 win is a loss for P2 and vice versa

            # Inf propagation

            # Climb up the tree
            _node = _node.parent


def ucb1_func(node: Node, c: float) -> Node:
    """
    Returns the most urgent child to visit using the Upper Bound Confidence interval
    :param node:
        Node: current node in which to check for most urgent child
    :param c:
        float: exploration parameter

    :return:
        Node: node of the most urgent child to visit
    """
    # Use UCB1 to select the next node to visit
    child_action_key = None
    ucb1_max = -np.inf
    for a in node.children.keys():
        child = node.children[a]
        ucb1 = (child.r / child.n) + c*np.sqrt((2*np.log(node.n))/child.n)

        # Select node with highest UCB1
        if ucb1 >= ucb1_max:
            ucb1_max = ucb1  # update max UCB1
            child_action_key = a

    return node.children[child_action_key], PlayerAction(child_action_key)


# What actions are available from a specific node
def poss_actions(s: np.ndarray) -> np.ndarray:
    """
    Return array with possible actions given the current state
    :param s:
        np.ndarray: current state matrix
    :return:
        np.ndarray: array with possible actions
    """
    return np.arange(s.shape[1], dtype=PlayerAction)[s[-1, :] == NO_PLAYER]
