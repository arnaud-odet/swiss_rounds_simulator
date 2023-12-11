import pandas as pd 
import string
from swiss_rounds_simulator.utils import simulate_n_tournaments

def run_simulations(n_tournaments:int, n_teams:int, n_rounds:int, possible_strategies = [], verbose = True):
    
    # Step 1 : naming the teams 
    alphabet = list(string.ascii_lowercase)
    n_rerun = n_teams // 26
    for i in range(n_rerun):
        alphabet = alphabet + [alphabet[i] + elem for elem in  alphabet]
    teams = alphabet[:n_teams]
    
    for team in teams :
        strategies = [{}]
        for possible_strategy in possible_strategies:
            strategies.append({team:possible_strategy})
            
        for strategy in strategies :
            deterministic_tmp_df = simulate_n_tournaments(n_tournaments=n_tournaments, nb_teams=n_teams, nb_games=n_rounds, strategies=strategy, method = 'deterministic')
            proba_linear_tmp_df = simulate_n_tournaments(n_tournaments=n_tournaments, nb_teams=n_teams, nb_games=n_rounds, strategies=strategy, method = 'probabilistic', delta_level='linear')
            proba_exponential_tmp_df = simulate_n_tournaments(n_tournaments=n_tournaments, nb_teams=n_teams, nb_games=n_rounds, strategies=strategy, method = 'probabilistic', delta_level='exponential')