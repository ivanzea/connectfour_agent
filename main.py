from typing import Optional, Callable

import numpy as np

from agents.agent_random import generate_move as gm_rand
from agents.agent_mcts.mcts_optim import generate_move_mcts as gm_mcts
from agents.agent_minimax import generate_move as gm_minimax
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove

def user_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]):
    action = PlayerAction(-1)
    while not 0 <= action < board.shape[1]:
        try:
            action = PlayerAction(input("Column? "))
        except:
            pass
    return action, saved_state

def agent_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove,
    encounters: int,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):
    import time
    from agents.common import PLAYER1, PLAYER2, GameState
    from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)
    for play_first in np.repeat((1, -1), encounters):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {"O" if player == PLAYER1 else "X"}'
                )

                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args)

                print(f"{player_name} move time: {time.time() - t0:.3f}s")
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                    else:
                        print(
                            f'{player_name} won playing {"O" if player == PLAYER1 else "X"}'
                        )
                    playing = False
                    break

if __name__ == "__main__":
    agent_vs_agent(gm_mcts, user_move, player_1='mcts', player_2='human', encounters=1)
