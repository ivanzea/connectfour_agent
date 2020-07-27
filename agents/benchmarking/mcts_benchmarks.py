"""
mcts_benchmarks.py:
    Performance tuning
"""
# Import packages
import numpy as np
import agents.common as cc
from agents.common import PlayerAction, BoardPiece
from agents.common import NO_PLAYER, PLAYER1, PLAYER2
from agents.agent_minimax import generate_move as minimax_move

from joblib import Parallel, delayed
import multiprocessing

import os
from typing import List

# Parameters ===========================================================================================================
from agents.agent_mcts.mcts_optim import generate_move_mcts as mcts_move  # what agent version?

results_path = '/mnt/d/GitHub/connectfour_agent/agents/benchmarking/results/'
output_file = 'selfplay_benchmarks'

# Node search speed
nss = False
sample_size = 1200

# Win rate
wr = False
number_games = 24
c = 1.5  # c = [0.1, 0.5, 1.0, 1.5*, 2.0, 2.5, 5.0, 10.0]

# Profiling
prof = False

# Define functions =====================================================================================================
# Check how many simulation it performs per second
def node_search_speed(dummy, max_t=1.0) -> float:
    """
    Performs mcts and check how many times it visited the root node in 1 second
    :param dummy:
        Used for parallelization
    :param max_t:
        float: amount of time to run mcts for before asking for an answer

    :return:
        float: number of times the root node was visited
    """
    # Initialize
    board = cc.initialize_game_state()

    # Start mcts
    _, ss = mcts_move(board=board, player=PLAYER1, saved_state=None, max_t=max_t)

    return sum([ss.root.children[key].n for key in ss.root.children.keys()])

# Asses node search speed in parallel
def visits_per_sec_benchmark(n: int, filename: str):
    """
    Benchmarking for number of simulations per second
    :param n:
        int: number of simulations to perform
    :param filename:
        str: how to save the simulation data

    :return:
        csv file in results directory in benchmarking folder
    """
    # Query node speed in parallel
    results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(node_search_speed)(i) for i in range(n))

    # Write output file
    np.savetxt(filename, results, delimiter=',')


# Have a games against minimax agent to check for winning rate
def agent_duel(dummy, agent_a, agent_b, a_name: str, b_name: str, c=1.0) -> List:
    """
    Check if agent_a (agent to be benchmarked) wins a duel as Player
    :param dummy:
        used for parallelization purposes
    :param agent_a:
        generate_move object for agent to be monitored
    :param agent_b:
        generate_move object for the agent playing as the opponent
    :param a_name:
        str: name tag of agent_a
    :param b_name:
        str: name tag of agent_b
    :param c:
        float: exploration coefficient

    :return:
        List: list with 2 values, each denoting a win or lose playing as P1 and P2 -> (P1, P2)
    """
    # Initialize output
    out = [0, 0]

    # Have a game with P1 and one with P2 starting
    for players in  [[PLAYER1, PLAYER2], [PLAYER2, PLAYER1]]:
        # Initialize game
        board = cc.initialize_game_state()
        agents = [agent_a, agent_b]
        names = [a_name, b_name]
        states = [None, None]

        # Play the game until end state is reached
        while cc.check_end_state(board=board, player=players[1]) == cc.GameState.STILL_PLAYING:
            # Choose an action
            if agents[0] == agent_a:
                action, states[0] = agents[0](board=board, player=players[0], saved_state=states[0], c=c)
            else:
                action, states[0] = agents[0](board=board, player=players[0], saved_state=states[0])

            # Apply action
            cc.apply_player_action(board=board, action=action, player=players[0])

            # Change player
            players = players[::-1]
            agents = agents[::-1]
            names = names[::-1]
            states = states[::-1]

        # Check who won
        if cc.check_end_state(board=board, player=players[1]) == cc.GameState.IS_WIN:
            result = 1 if names[1] == a_name else 0

            # Won as P1 or P2
            player_index = 0 if players[1] == PLAYER1 else 1
            out[player_index] += result

    return out

def winning_rate(n: int,  filename: str, c=1.0):
    """
    Approximate a winning for the mcts agent against minimax
    :param n:
        int: number of games to play
    :param filename:
        str: how to save the simulation data
    :param c:
        float: exploration coefficient

    :return:
        csv file with wins and loses
    """
    # Query node speed in parallel
    results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(agent_duel)(i, agent_a=mcts_move,
                       agent_b=mcts_move, a_name='mcts', b_name='mcts2', c=c) for i in range(n))

    # Write output file
    np.savetxt(filename, results, delimiter=',')


# Benchmark execution ==================================================================================================
# Node search speed
if nss:
    visits_per_sec_benchmark(n=sample_size, filename=results_path+output_file+'_nss.csv')

# Win rate
if wr:
    fname = results_path+output_file+f'_c{c}_wr.csv'
    winning_rate(n=number_games, filename=fname, c=c)

# Profiling
if prof:
    import cProfile
    cProfile.run('node_search_speed(1, 60)', results_path+output_file+'_profile.prof')
