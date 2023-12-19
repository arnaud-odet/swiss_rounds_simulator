import pandas as pd 
import numpy as np
import string
import os
from swiss_rounds_simulator.utils import simulate_n_tournaments, ray_simulate_n_tournaments

def run_simulations(n_tournaments:int, n_teams:int, n_rounds:int, thresholds:list=[],  possible_strategies = [], method = 'probabilistic', delta_level='linear', parallelize= False, verbose = True, verbose_prompt=''):
    
    n_simu_tot = n_tournaments * (n_teams * len(possible_strategies) +1)
    n_simu_done = 0
    
    # Step 1 : naming the teams 
    alphabet = list(string.ascii_lowercase)
    n_rerun = n_teams // 26
    for i in range(n_rerun):
        alphabet = alphabet + [alphabet[i] + elem for elem in  alphabet]
    teams = alphabet[:n_teams]
    
    # Step 2 : runing the control simulations
    if parallelize:
        df = ray_simulate_n_tournaments(n_tournaments=n_tournaments, nb_teams=n_teams, nb_games=n_rounds, thresholds=thresholds ,strategies={}, method = method, delta_level=delta_level)
    else :
        df = simulate_n_tournaments(n_tournaments=n_tournaments, nb_teams=n_teams, nb_games=n_rounds, thresholds=thresholds ,strategies={}, method = method, delta_level=delta_level)
    n_simu_done += n_tournaments
    if verbose:
        print(f"{verbose_prompt} - Number of simulations to perform : {n_simu_tot} - Number of simulations performed : {n_simu_done} ({np.round(100*n_simu_done/n_simu_tot,1)}%) ")
    
    # Step 3 : preparing the DF to be merged
    df.drop(columns = ['Strategy'], inplace = True)
    control_rename_dict = {col_name: 'control_'+ col_name for col_name in df.columns}
    control_rename_dict['Id'] = 'Id'
    control_rename_dict['Level'] = 'Level'
    df.rename(columns = control_rename_dict, inplace = True)  
      
    cols_to_duplicate = ['Avg_WR','Avg_Rank'] + ['Thres_' + str(k) for k in thresholds]
    new_cols = ['S_L' + str(j) + '_' + col for j in possible_strategies for col in  cols_to_duplicate]
    for col in new_cols:
        df[col] = 0.0
    
    # Step 4 : iterating over teams and strategies
    for team in teams :
        strategies = []
        for possible_strategy in possible_strategies:
            strategies.append({team:list(range(1,possible_strategy+1))})
            
        for strat_id, strategy in zip(possible_strategies,strategies) : 
            col_id = 'S_L' + str(strat_id) + '_'
            if parallelize:
                tmp_df = ray_simulate_n_tournaments(n_tournaments=n_tournaments, nb_teams=n_teams, nb_games=n_rounds, thresholds=thresholds ,strategies=strategy, method = method, delta_level=delta_level)
            else :
                tmp_df = simulate_n_tournaments(n_tournaments=n_tournaments, nb_teams=n_teams, nb_games=n_rounds, thresholds=thresholds ,strategies=strategy, method = method, delta_level=delta_level)
            for col in cols_to_duplicate :
                df.loc[team, col_id+col] = tmp_df.loc[team, col]
            n_simu_done += n_tournaments
            if verbose:
                print(f"{verbose_prompt} - Number of simulations to perform : {n_simu_tot} - Number of simulations performed : {n_simu_done} ({np.round(100*n_simu_done/n_simu_tot,1)}%) ")
    
    return df.sort_values(by='Id')

if __name__ == '__main__':
    
    filepath = os.getcwd() + '/data/'
    
    nb_tourn = 100
    poss_nb_teams = [16,18,36]
    poss_nb_games = [6,6,8]
    poss_thresholds = [[8],[4,12],[8,24]]
    possible_strats = [1,2,3]
    setups = [('probabilistic','linear','prob_lin'),('probabilistic','exponential', 'prob_exp'),('deterministic','linear','deterministic')]
    
    nb_batchs = len(poss_nb_teams) * len(setups)
    
    for k, nb_teams, nb_games, thresholds in zip(range(nb_batchs), poss_nb_teams, poss_nb_games, poss_thresholds):
    
        for i,setup in enumerate(setups) :
            filename = filepath + 'S=' + str(nb_tourn) + '_T=' + str(nb_teams) + '_R=' + str(nb_games) + '_' + setup[2] 
            prompt_str = f"Batch {k * len(setups) + i + 1} out of {nb_batchs} ({np.round(100*   (k*len(setups)+i)/nb_batchs,1)}% completed) - Teams : {nb_teams}, rounds : {nb_games}, set-up : {setup[2]} "
            out = run_simulations(
                n_tournaments=nb_tourn,
                n_teams=nb_teams,
                n_rounds=nb_games,
                thresholds=thresholds,
                possible_strategies=possible_strats,
                method=setup[0],
                parallelize=True,
                delta_level=setup[1],
                verbose_prompt=prompt_str
            )
            out.to_csv(filename+'.csv',index_label='team')
    