"""
minimax.py
    Implementation of a connect 4 agent that plays using the minimax algorithm
"""
# Import packages
import numpy as np
from numba import njit
import agents.common as cc
from agents.common import BoardPiece, SavedState, PlayerAction
from agents.common import PLAYER1, PLAYER2, NO_PLAYER
from typing import Tuple, Optional


def generate_move_minimax(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose an action based on the minimax algorithm
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: player piece to check for best move
    :param saved_state:
        SavedState: standard generate_move input... not used in this case
    :return:
        PlayerAction: column to play based on the minimax algorithm
    """
    # Apply minimax algorithm
    action = minimax(board=board, player=player)

    return action, saved_state


def minimax(board: np.ndarray, player: BoardPiece):
    """
    Minimax algorithm
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: player piece to evaluate for best move
    :return:
        PlayerAction: best possible action
    """
    # Score the board and player possible actions
    poss_actions = cc.possible_actions(board=board)
    poss_actions = poss_actions[np.argsort(np.abs(poss_actions-3))]  # center search bias
    action = PlayerAction(poss_actions[0])
    score = board_score(board=board, player=player)

    for moves in poss_actions:
        # How would a mover change my score?
        move_board = cc.apply_player_action(board=board, action=moves, player=player, copy=True)
        move_score = board_score(board=move_board, player=player)
        #print(f'{moves} = {move_score}')
        # Does it result in a better value?
        if move_score > score:
            #print(f'{moves} -> {move_score} > {score}')
            # Update variables
            score = move_score
            action = moves

    return action


@njit()
def board_score(board: np.ndarray, player: BoardPiece) -> float:
    """
    Apply a heuristic scoring algorithm for a game board
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: type of Player piece to check for score
    :return:
        float: total state of the board heuristic score
    """
    # Number of consecutive Player pieces considered winning condition
    connect_length = [2, 3, 4]
    score_dict = [10, 50, 1000]

    # Initialize score
    score = 0

    for connect_n, dict_val in zip(connect_length, score_dict):
        # Initialize variables
        rows, cols = board.shape
        rows_edge = rows - connect_n + 1
        cols_edge = cols - connect_n + 1

        # Horizontal scoring
        for i in range(rows):
            for j in range(cols_edge):
                if np.all(board[i, j:j + connect_n] == player):
                    score += dict_val

        # Vertical scoring
        for i in range(rows_edge):
            for j in range(cols):
                if np.all(board[i:i + connect_n, j] == player):
                    score += dict_val

        # Diagonal scoring
        for i in range(rows_edge):
            for j in range(cols_edge):
                block = board[i:i + connect_n, j:j + connect_n]
                # Diagonal R
                if np.all(np.diag(block) == player):
                    score += dict_val
                # Diagonal L
                if np.all(np.diag(block[::-1, :]) == player):
                    score += dict_val
    return score
