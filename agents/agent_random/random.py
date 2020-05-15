# Import packages
import numpy as np
import agents.common as cc
from agents.common import BoardPiece, SavedState, PlayerAction
from agents.common import PLAYER1, PLAYER2, NO_PLAYER
from typing import Tuple, Optional


# Randomly generate a possible action
def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    action = PlayerAction(np.random.choice(cc.possible_actions(board)))

    return action, saved_state
