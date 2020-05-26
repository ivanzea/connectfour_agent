"""
minimax.py
    Implementation of a connect 4 agent that plays using the minimax algorithm
"""
# Import packages
from typing import Tuple, Optional, List

import numpy as np
from numba import njit

import agents.common as cc
from agents.common import BoardPiece, SavedState, PlayerAction
from agents.common import PLAYER1, PLAYER2, NO_PLAYER


def generate_move_minimax(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], score_dict = np.array([50, 50, 10, 10, 5])
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose an action based on the minimax algorithm
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: player piece to check for best move
    :param score_dict:
        np.ndarray: list of score points to give to the different patterns... see board_score
    :param saved_state:
        SavedState: standard generate_move input... not used in this case
    :return:
        PlayerAction: column to play based on the minimax algorithm
    """
    # Apply minimax algorithm
    action, _ = minimax(board=board, player=player, score_dict=score_dict, depth=6,
                        alpha=-np.infty, beta=np.infty, maxplayer=True)

    return action, saved_state


def minimax(board: np.ndarray, player: BoardPiece, score_dict: np.ndarray,
            depth: int, alpha: float, beta: float, maxplayer: bool) -> (PlayerAction, float):
    """
    Minimax algorithm with alpha-beta pruning
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: player piece to evaluate for best move (maximazing player)
    :param score_dict:
        np.ndarray: list of score points to give to the different patterns... see board_score
    :param depth:
        int: depth of tree search
    :param alpha:
        float: keep track of best score
    :param beta:
        float: keep track of worst score
    :param maxplayer:
        bool: flag if the maximizing player is playing
    :return:
        (PlayerAction, float): best possible action and its score
    """
    # Get possible moves
    # Player possible actions
    poss_actions = (np.arange(board.shape[1])[board[-1, :] ==  NO_PLAYER])
    poss_actions = poss_actions[np.argsort(np.abs(poss_actions - 3))]  # center search bias
    pieces = np.array([PLAYER1, PLAYER2])

    # Final or end state node reached
    if (depth == 0) | (cc.check_end_state(board=board, player=player) != cc.GameState.STILL_PLAYING):
        if (cc.check_end_state(board=board, player=player) == cc.GameState.IS_WIN) & maxplayer:
            return None, 1000
        if (cc.check_end_state(board=board, player=player) == cc.GameState.IS_WIN) & ~maxplayer:
            return None, -1000
        if cc.check_end_state(board=board, player=player) == cc.GameState.IS_DRAW:
            return None, 0
        else:
            return None, board_score(board=board, player=player, score_dict=score_dict)

    if maxplayer:
        # Initialize score
        max_score = -np.infty

        for moves in poss_actions:
            # How would a mover change my score?
            move_board = cc.apply_player_action(board=board, action=moves, player=player, copy=True)
            score = minimax(board=move_board, player=player, score_dict=score_dict, depth=depth-1,
                            alpha=alpha, beta=beta, maxplayer=False)[1]

            if score > max_score:
                max_score = score
                action = moves
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return action, max_score
    else:
        # Initialize opponent score
        min_score = np.infty
        opponent = pieces[pieces != player][0]

        for moves in poss_actions:
            # How would a mover change my score?
            move_board = cc.apply_player_action(board=board, action=moves, player=opponent, copy=True)
            score = minimax(board=move_board, player=opponent, score_dict=score_dict, depth=depth - 1,
                            alpha=alpha, beta=beta, maxplayer=True)[1]

            if score < min_score:
                min_score = score
                action = moves
            beta = min(beta, score)
        return action, min_score


@njit()
def board_score(board: np.ndarray, player: BoardPiece, score_dict: np.ndarray) -> float:
    """
    Apply a heuristic scoring algorithm for a game board
    :param board:
        np.ndarray: current state of the board, filled with Player pieces
    :param player:
        BoardPiece: type of Player piece to check for score
    :param score_dict:
        np.ndarray: list of score points to give to the different patterns... see board_score
    :return:
        float: total state of the board heuristic score
    """
    # Number of consecutive Player pieces considered winning condition
    p = player
    n = NO_PLAYER
    connect_patterns = np.array([[p, p, p, n],
                                 [p, p, n, p],
                                 [p, p, n, n],
                                 [p, n, p, n],
                                 [p, n, n, p]])
    connect_n = 4

    # Initialize score
    score = 0

    for pattern, dict_val in zip(connect_patterns, score_dict):
        # Initialize variables
        rows, cols = board.shape
        rows_edge = rows - connect_n + 1
        cols_edge = cols - connect_n + 1

        # Horizontal scoring
        for i in range(rows):
            for j in range(cols_edge):
                if np.all(board[i, j:j + connect_n] == pattern) or np.all(board[i, j:j + connect_n] == pattern[::-1]):
                    score += dict_val

        # Vertical scoring
        for i in range(rows_edge):
            for j in range(cols):
                if np.all(board[i:i + connect_n, j] == pattern) or np.all(board[i:i + connect_n, j] == pattern[::-1]):
                    score += dict_val

        # Diagonal scoring
        for i in range(rows_edge):
            for j in range(cols_edge):
                block = board[i:i + connect_n, j:j + connect_n]
                # Diagonal
                if np.all(np.diag(block) == pattern) or np.all(np.diag(block) == pattern[::-1]):
                    score += dict_val
                # Diagonal L
                if np.all(np.diag(block[::-1, :]) == pattern) or np.all(np.diag(block[::-1, :]) == pattern[::-1]):
                    score += dict_val
    return score
