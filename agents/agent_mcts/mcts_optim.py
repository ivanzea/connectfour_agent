"""
mcts_base.py
    Implementation of a connect 4 agent that plays using the monte carlo tree search algorithm
"""
# Import packages
import time

from typing import Tuple, Optional

import numpy as np
from math import  log
from random import choice

import agents.common as cc
from agents.common import BoardPiece, SavedState, PlayerAction
from agents.common import PLAYER1, PLAYER2, NO_PLAYER


def generate_move_mcts(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], c=1.5, max_t=4.99,
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose an action based on the mcts algorithm
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: player piece to check for best move
    :param c:
        float: exploration/explotation coefficient
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
            if (node.s == board).all():
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

        # End state values
        if self.status == cc.GameState.IS_WIN:
            self.r = np.inf

    def next_node(self, c: float):
        """
        Selects the next node to simulate a game from using either expansion or ucb1 to find the most
        urgent child node
        :param c:
            float: exploration/exploitation parameter

        :return:
            Node: child node to simulate a game from
        """
        # Define the current node
        _node = self

        # Repeat until terminal node is found
        while _node.status == cc.GameState.STILL_PLAYING:
            # Apply tree search policy
            poss_actions = np.arange(_node.s.shape[1], dtype=PlayerAction)[_node.s[-1, :] == NO_PLAYER]
            actions = [act for act in poss_actions if not act in list(_node.children.keys())]  # find possible unexplored actions v1.0

            # Check for unexplored children nodes
            if len(actions) != 0:
                # Take a random actions
                a = choice(actions)  # select action

                # Add new child node
                _node.children[a] = Node(state=cc.apply_player_action(_node.s, a, _node.next_player, copy=True),
                                         player=_node.next_player, parent=_node)

                return _node.children[a]
            else:  # explore which is the best child using UBC1
                _node, _ = ucb1_func(_node, c)

        return _node

    def simulation(self):
        """
        Simulate a game in a random fashion until a end node and respective reward is found
        :return:
            float: resulting reward from the simulated random game
        """
        # Take random actions until an end state is reached
        player = self.player
        opponent = self.next_player

        # Perform simulation until end state is reached
        s = self.s.copy()
        players = [player, opponent]
        while cc.check_end_state(s, players[0]) == cc.GameState.STILL_PLAYING:
            # Take a random action
            poss_actions = np.arange(s.shape[1], dtype=PlayerAction)[s[-1, :] == NO_PLAYER]
            action = choice(poss_actions)
            i = sum(s[:, action] != NO_PLAYER)
            s[i, action] = players[1]

            # Change player
            players = players[::-1]

        # Check who won the match
        if cc.check_end_state(s, players[0]) == cc.GameState.IS_WIN:
            return 1 if players[0] == player else -1  # if the winner is equal to the initial player, r = 1
        else:
            return 0.0  # case of a tie


    def backprop(self, delta):
        """
        Back propagate the reward calculated from the game simulation. Also, apply the special case
        of terminal node back propagation for node pruning
        :param delta:
            float: reward to be back propagated
        """
        _node = self

        # Check for terminal node and use special case back propagation
        if _node.r == np.inf:
            # Inf value backprop
            inf_rule = True  # check thant propagation rules still apply

            while inf_rule and (None != _node.parent):
                # Update n
                _node.n += 1

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
        float: exploration/exploitation parameter

    :return:
        Node: node of the most urgent child to visit
    """
    # Initialize variables
    child_action_key, urgent_child = None, None
    ucb1_max = -np.inf

    # Use UCB1 to select the next node to visit
    for a, child in zip(node.children.keys(), node.children.values()):
        ucb1 = (child.r / child.n) + c * (((2 * log(node.n)) / child.n) ** (1 / 2))

        # Select node with highest UCB1
        if ucb1 >= ucb1_max:
            ucb1_max = ucb1  # update max UCB1
            urgent_child = child
            child_action_key = a

    return urgent_child, PlayerAction(child_action_key)
