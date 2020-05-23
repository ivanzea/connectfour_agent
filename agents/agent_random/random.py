"""
random.py
    Implementation of a connect 4 agent that plays randomly
"""
# Import packages
import numpy as np
import agents.common as cc
from agents.common import BoardPiece, SavedState, PlayerAction
from agents.common import PLAYER1, PLAYER2, NO_PLAYER
from typing import Tuple, Optional


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
   Choose randomly an action based on available free columns
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: standard generate_move input... not used in this case
    :param saved_state:
        SavedState: standard generate_move input... not used in this case
    :return:
        PlayerAction: random column to play
    """
    # Choose a valid, non-full column randomly and return it as `action`
    action = PlayerAction(np.random.choice(np.arange(board.shape[1])[board[-1, :] ==  NO_PLAYER]))

    return action, saved_state
