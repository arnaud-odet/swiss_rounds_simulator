import pandas as pd
import numpy as np
import string
import random
from swiss_rounds_simulator.mwmatching import maxWeightMatching



def initiate_league(n_teams, n_rounds, delta_level='linear' ,strategies = {}):
    if n_teams % 2 == 1 :
        raise ValueError('Please enter an even number of teams')
    if n_rounds > n_teams - 1 :
        raise ValueError('For simplicity purpose, we limit the analysis to competition where teams play each other at most once')
    if n_teams > 702 :
        raise ValueError('For identification prupose, the number of teams is limited to 702 (to be named from a to zz)')    

    alphabet = list(string.ascii_lowercase)
    n_rerun = n_teams // 26
    for i in range(n_rerun):
        alphabet = alphabet + [alphabet[i] + elem for elem in  alphabet]

    teams = alphabet[:n_teams]
    tm_nbrs = range(n_teams)
    
    calendar_columns = ['Id','Level','Strategy','Nb_win','Nb_loss',"Win_rate","OWR"]
    for i in range(n_rounds):
        calendar_columns.append(f"R{i+1}_opponent")
        calendar_columns.append(f"R{i+1}_result")
    
    levels = []
    if delta_level == 'linear':
        for j in range(n_teams):
            levels.append((n_teams-j)/n_teams)
    elif delta_level =='exponential':
        for j in range(n_teams):
            levels.append( (1 / 2)**j )
    init_cal = [tm_nbrs] + [levels] + [[[]]*n_teams] + [[0]*n_teams] * 4 + ([['-']*n_teams] * (2*n_rounds))
    league_table = pd.DataFrame(init_cal, index= calendar_columns, columns = teams).transpose()
    
    for strat in strategies.keys():
        try :
            league_table.loc[strat,'Strategy'] = strategies[strat]
        except : 
            pass
    
    return league_table.round(2)


def assign_opponents(league_table:pd.DataFrame, round_number:int, verbose = True):

    if round_number == 1:
        league_table = league_table.sample(frac = 1) 
    edges = []
    for a in league_table.index:
        id_a = league_table.loc[a,'Id']
        a_past_opp = []
        for i in range(round_number-1):
            a_past_opp.append(league_table.loc[a,f"R{i+1}_opponent"])
        for b in league_table.index:
            id_b = league_table.loc[b,'Id']
            if a == b :
                pass
            elif b in a_past_opp :
                edges.append((id_a,id_b,0))
            else:
                edges.append((id_a,id_b,10000 - int(np.abs(league_table.loc[a,'Win_rate'] - league_table.loc[b,'Win_rate'])*100)**2)) 
                
    pairing = maxWeightMatching(edges, maxcardinality=True)

    for team_id in range(len(pairing)):
        team = list(league_table.query(f"Id == {team_id}").index)[0]
        opponent = list(league_table.query(f"Id == {pairing[team_id]}").index)[0]
        if league_table.loc[team,f"R{round_number}_opponent"] == '-':
            league_table.loc[team,f"R{round_number}_opponent"] = opponent
            league_table.loc[opponent,f"R{round_number}_opponent"] = team
            if verbose :
                team_record_str = f"({league_table.loc[team,'Nb_win']}-{league_table.loc[team,'Nb_loss']})"
                opponent_record_str = f"({league_table.loc[opponent,'Nb_win']}-{league_table.loc[opponent,'Nb_loss']})"
                print(f"Round number {round_number} : Team {team} {team_record_str} was paired with team {opponent} {opponent_record_str}")
        
    return league_table


def simulate_game(team_level, opponent_level, team_strat = 0, opponent_strat = 0, method = 'probabilistic', verbose = False):
    """
    Method can be either 'probabilistic' or 'deterministic'
    Team_strat (respectively opponent_strat) takes 1 if the team (respectively the opponent) choose to purposedly loose the game
    """
    
    if team_strat == 1 and opponent_strat == 0 :
        res = ("Loss", "Win")
    elif team_strat == 0 and opponent_strat == 1 :
        res = ("Win", "Loss")
    else :
        if method == 'deterministic' and not team_level == opponent_level:
            if team_level > opponent_level:
                res = ("Win", "Loss")
            else :
                res = ("Loss", "Win")
        else :
            threshold = team_level / (opponent_level + team_level)
            rand = np.random.random()   
            if rand < threshold : 
                res = ("Win", "Loss")   
            else : 
                res = ("Loss", "Win")
    return res


def play_round(league_table, round_number, method = 'probabilistic', verbose = True):
    
    if f"R{round_number}_opponent" not in league_table.columns: 
        raise ValueError(f'The league was initiated with less than {round_number} games')
    
    opp_str = f'R{round_number}_opponent'
    res_str = f'R{round_number}_result'
    for team in league_table.index:
        if league_table.loc[team, res_str] == '-':
            opponent = league_table.loc[team, opp_str]            
            if verbose :
                # keeping memory of pre-game records
                team_record_str = f"({league_table.loc[team,'Nb_win']}-{league_table.loc[team,'Nb_loss']})"
                opponent_record_str = f"({league_table.loc[opponent,'Nb_win']}-{league_table.loc[opponent,'Nb_loss']})"
            team_level = league_table.loc[team, 'Level']
            opponent_level = league_table.loc[opponent, 'Level']
            team_strat_list = league_table.loc[team, 'Strategy']
            opponent_strat_list = league_table.loc[opponent, 'Strategy']
            team_strat = 1 if round_number in team_strat_list else 0
            opponent_strat = 1 if round_number in opponent_strat_list else 0
            strat = team_strat + opponent_strat
            result = simulate_game(team_level, opponent_level, team_strat, opponent_strat, method = method, verbose = verbose)
            league_table.loc[team,res_str] = result[0]
            league_table.loc[opponent,res_str] = result[1]
            if result[0] == 'Win':
                league_table.loc[team,'Nb_win'] +=1
                league_table.loc[opponent,'Nb_loss'] +=1
                winner = team
            else :
                league_table.loc[opponent,'Nb_win'] +=1
                league_table.loc[team,'Nb_loss'] +=1
                winner = opponent           
            if (team_level > opponent_level and result[0] == 'Loss') or team_level < opponent_level and result[0] == 'Win':
                unexpected = True
            else :
                unexpected = False
            
            if verbose :
                print(f'Round number {round_number}, game {team} {team_record_str} vs {opponent} {opponent_record_str} of levels {np.round(team_level,2)} vs {np.round(opponent_level,2)}: victory for {winner}{" - UNEXPECTED" if unexpected else ""} {" - STRATEGIC " if strat > 0 else ""}')

    league_table['Win_rate'] = league_table['Nb_win'] / round_number
    for team in league_table.index :
        opponents_wr = []
        for i in range(round_number):
            opponents_wr.append(league_table.loc[league_table.loc[team, f"R{i+1}_opponent"],'Win_rate'])
        league_table.loc[team,'OWR'] = sum(opponents_wr) / len(opponents_wr)
    
    return league_table.sample(frac=1).sort_values(by=['Win_rate','OWR'], ascending = False)  


def rank_table(league_table:pd.DataFrame, n_rounds:int, verbose=True):
    if 'Rank' not in league_table.columns:
        league_table['Rank'] = [0] * league_table.shape[0]

    init_ranks = league_table['Rank']
    keep_ranking=True
    use_OWR = False
    j = 1
    
    while keep_ranking:
        multi_given_ranks = init_ranks.value_counts()[init_ranks.value_counts()>1].index     
        n_unranked_init = init_ranks.value_counts()[init_ranks.value_counts()>1].sum()
        if verbose :
            print(f"Iteration number {j}, {use_OWR=}, number of unranked teams = {n_unranked_init} ")
        for initial_rank in multi_given_ranks :
            subcomp_df = league_table.query(f"Rank =={initial_rank}")
            subcomps = []
            for score in subcomp_df['Nb_win'].unique():
                subtable = subcomp_df.query(f"Nb_win == {score}")
                for team in subtable.index:
                    team_subcomp_games = []
                    for i in range(n_rounds):
                        opponent = subcomp_df.loc[team,f'R{i+1}_opponent']
                        if opponent in subtable.index:
                            result = subcomp_df.loc[team,f'R{i+1}_result']
                            team_subcomp_games.append(1 if result == 'Win' else 0)
                    team_subscore = np.mean(team_subcomp_games) if len(team_subcomp_games) >0 else 0.5
                    subcomps.append({'team':team, 'subscore':team_subscore})
            tmp_df = pd.DataFrame(subcomps).set_index('team')
            subcomp_df = subcomp_df.merge(tmp_df, left_index=True, right_index=True)
            subcomp_df['tmp_rank'] = 1000000 * subcomp_df['Nb_win'] + 1000 * subcomp_df['subscore'] + (subcomp_df['OWR'] if use_OWR else 0)
            if verbose :
                print('Working on subgroup')    
                display(subcomp_df)
            r_df = subcomp_df['tmp_rank'].sort_values(ascending=False).drop_duplicates().reset_index().drop(columns = 'index').reset_index().set_index('tmp_rank')
            for team in subcomp_df.index:   
                league_table.loc[team,'Rank'] = r_df.loc[subcomp_df.loc[team,'tmp_rank'],'index'] + initial_rank
        # re-rank
        re_rank_dict = {}
        for rank in league_table['Rank']:
            rank_restated = league_table.query(f"Rank < {rank}").shape[0]
            re_rank_dict[rank] = rank_restated
        league_table['Rank'] = league_table['Rank'].map(re_rank_dict)
        
        final_ranks = league_table['Rank']
        n_unranked_final = final_ranks.value_counts()[final_ranks.value_counts()>1].sum()
        if n_unranked_init == n_unranked_final:
            if verbose :
                print("Test on ranks = True")
            if use_OWR :
                keep_ranking = False
            else :
                use_OWR = True
                init_ranks = final_ranks    
                j+=1
        else :
            if verbose :
                print("Test on ranks = False")
            init_ranks = final_ranks    
            j+=1

    league_table.sort_values(by='Rank', inplace = True)
    league_table['Rank'] = league_table['Rank'] +1
    league_table['Win_rate'] = league_table['Win_rate'].apply(lambda x:np.round(x,2))
    league_table['OWR'] = league_table['OWR'].apply(lambda x:np.round(x,2))
    return league_table


def simulate_tournament(nb_teams, nb_games, strategies = {}, method = 'probabilistic', delta_level = 'linear', verbose = True):

    lt = initiate_league(nb_teams,nb_games, delta_level=delta_level,strategies=strategies)
    for i in range(nb_games):
        lt = assign_opponents(lt,i+1, verbose = False)
        lt = play_round(lt,i+1, method = method,verbose = verbose)
    lt = rank_table(lt,nb_games, verbose = False)
    return lt


def simulate_n_tournaments(n_tournaments, nb_teams, nb_games, strategies = {}, method = 'probabilistic', delta_level = 'linear') :

    first = True 
    
    for i in range(n_tournaments) :
        lt = simulate_tournament(nb_teams, nb_games, strategies = strategies, delta_level=delta_level,method = method, verbose = False)
        if first :
            wr_df = lt[['Level','Strategy','Win_rate']].rename(columns={'Win_rate':'WR_0'})
            rk_df = lt[['Rank']].rename(columns={'Rank':'Rank_0'})
            first = False
        else :
            temp_wr = lt[['Win_rate']].rename(columns={'Win_rate':f'WR_{i}'})
            temp_rk = lt[['Rank']].rename(columns={'Rank':f'Rank_{i}'})
            wr_df = wr_df.merge(temp_wr, left_index=True, right_index=True)
            rk_df = rk_df.merge(temp_rk, left_index=True, right_index=True)
    wr_df['Avg_WR'] = wr_df.drop(columns = ['Level','Strategy']).mean(axis=1)
    rk_df['Avg_Rank'] = rk_df.mean(axis=1)
    output = wr_df[['Level','Strategy','Avg_WR']].merge(rk_df[['Avg_Rank']], left_index=True, right_index=True)
    output['Level'] = output['Level'].apply(lambda x:np.round(x,2))
    return output.sort_values(by = 'Avg_Rank', ascending = True)


def compare_settings(n_tournaments, n_teams, n_rounds, delta_level='linear' ,strategies={}, probabilistic=True, deterministic=True):
    
    if probabilistic :
        print('Probabilistic resolution')
        r = simulate_n_tournaments(n_tournaments,n_teams,n_rounds, delta_level=delta_level, method = 'probabilistic')
        r.rename(columns = {'Avg_WR': 'Control_avg_WR', 'Avg_Rank': 'Control_avg_Rank'}, inplace = True)
        rs = simulate_n_tournaments(n_tournaments,n_teams,n_rounds, delta_level=delta_level, method = 'probabilistic', strategies = strategies)
        rs.rename(columns = {'Avg_WR': 'Strategic_avg_WR', 'Avg_Rank': 'Strategic_avg_Rank'}, inplace = True)
        rs = rs.merge(r[['Control_avg_WR', 'Control_avg_Rank']], left_index= True, right_index=True)
        rs['Delta_WR'] = rs['Strategic_avg_WR'] - rs['Control_avg_WR']
        rs['Delta_Rank'] = rs['Strategic_avg_Rank'] - rs['Control_avg_Rank']
        display(rs.round(2))
    
    if probabilistic and deterministic :
        print('---------------------------------------------------')
    
    if deterministic:
        print('Deterministic resolution')
        d = simulate_n_tournaments(n_tournaments,n_teams,n_rounds, delta_level=delta_level, method = 'deterministic')
        d.rename(columns = {'Avg_WR': 'Control_avg_WR', 'Avg_Rank': 'Control_avg_Rank'}, inplace = True)
        ds = simulate_n_tournaments(n_tournaments,n_teams,n_rounds, delta_level=delta_level, method = 'deterministic', strategies = strategies)
        ds.rename(columns = {'Avg_WR': 'Control_avg_WR', 'Avg_Rank': 'Control_avg_Rank'}, inplace = True)
        ds = ds.merge(d[['Control_avg_WR', 'Control_Avg_Rank']], left_index= True, right_index=True)
        ds['Delta_WR'] = ds['Strategic_avg_WR'] - ds['Control_avg_WR']
        ds['Delta_Rank'] = ds['Strategic_avg_Rank'] - ds['Control_avg_Rank']
        display(ds.round(2))  
        
    if not probabilistic and not deterministic :
        print('No setting were set to True for comparison')
        
        
def DEPRECATED_assign_opponents(possible_games_matrix, league_table, round_number, verbose = False):
    
    if f"R{round_number}_opponent" not in league_table.columns: 
        raise ValueError(f'The league was initiated with less than {round_number} games')
    
    ref_str = f'R{round_number}_opponent'
    
    # Preferred allocation based on rankings from top to bottom
    lt = league_table.copy()
    pgm = possible_games_matrix.copy()
    nb_success = 0
    if round_number == 1 :
        lt = lt.sample(frac = 1)
    else :
        lt.sort_values(by='Win_rate', ascending = False, inplace=True)
    
    for team in lt.index:
        if lt.loc[team, ref_str] == '-' :
            for opposing_team in lt.index:
                if opposing_team != team and pgm.loc[team,opposing_team] and lt.loc[opposing_team, ref_str] == '-' :
                    lt.loc[team, ref_str] = opposing_team
                    lt.loc[opposing_team, ref_str] = team
                    pgm.loc[team, opposing_team] = False
                    pgm.loc[opposing_team, team] = False
                    nb_success +=2
                    if verbose and nb_success == lt.shape[0] :
                        print(f'Round number {round_number} - Successfull allocation based on ranking')
                    break
    """
    if not nb_success == lt.shape[0] :
        # Second method based on distances in delta_win_rate
        lt = league_table.copy()
        pgm = possible_games_matrix.copy()
        nb_success = 0
        ### Determining possible opposition based on game_matrix
        pairs = []
        for a in lt.index :
            for b in lt.index :
                if pgm.loc[a,b]:
                    pairs.append({'pair' :[a,b], 'wr_delta' : np.abs(lt.loc[a,'Win_rate'] - lt.loc[b,'Win_rate'])})
        poss_pairs = pd.DataFrame(pairs)

        ### For all team, keep the bottom half delta_wr
        n_teams = len(list(lt.index))
        add_indices = []
        for team in lt.index:
            mask = [team in game for game in poss_pairs['pair']]
            team_indices = list(poss_pairs[mask].sort_values(by='wr_delta', ascending = True)[:int(poss_pairs.shape[0]/n_teams)].index)
            add_indices = add_indices + team_indices
            add_indices = list(dict.fromkeys(add_indices))
        poss_pairs = poss_pairs.loc[add_indices]

        ### Droping duplicates
        seen_pairs =[]
        drop_ind = []
        for ind,pair in zip(poss_pairs.index, poss_pairs['pair']):
            if [pair[1], pair[0]] in seen_pairs :
                drop_ind.append(ind)
            else : 
                seen_pairs.append(pair)
        poss_pairs = poss_pairs.drop(drop_ind)
        
        ### Finding all possible appairments
        pairs = list(poss_pairs['pair'])
        pairings = [[pair] for pair in pairs]
        for i in range(int(n_teams/2)-1):
            pps = []
            for pairing in pairings :
                for pair in pairs :
                    possible = True
                    for team in pair :
                        for game in pairing :
                            if team in game :
                                possible = False
                    if possible :
                        new_pairing = pairing + [pair]
                        pps.append(new_pairing) 
            pairings = pps.copy()  

        ### Selecting an random appairment over those with minimum Mean Absolute Delta WR
        deltas_wr = []
        for pp in pairings :
            delta_wr = 0
            for game in pp :
                delta_wr += np.abs(lt.loc[game[0],'Win_rate'] - lt.loc[game[1],'Win_rate'])
            deltas_wr.append(delta_wr)
        res = pd.DataFrame(pairings, deltas_wr).reset_index().rename(columns={'index':'mean_absolute_delta_wr'}).sort_values(by = 'mean_absolute_delta_wr',ascending=True)
        res = res[res['mean_absolute_delta_wr'] == res['mean_absolute_delta_wr'].min()].sample(frac=1).iloc[0]

        ### Allocating opponent based on res
        for i in range(int(n_teams/2)):
            try :
                lt.loc[res[i][0], ref_str] = res[i][1]
                lt.loc[res[i][1], ref_str] = res[i][0]
                pgm.loc[res[i][0], res[i][1]] = False
                pgm.loc[res[i][1], res[i][0]] = False
                nb_success += 2
            except : 
                pass
        if verbose and nb_success == lt.shape[0] :
            print(f'Round number {round_number} - Successfull allocation based on minimizing delta_wr distances')  
            return pgm, lt     
    """
           
    # Last resort : random allocation if none of the above worked
    if not nb_success == lt.shape[0] :
        keep_going = True
        while keep_going :
            nb_success = 0
            lt = league_table.copy()
            pgm = possible_games_matrix.copy()           
            lt = lt.sample(frac = 1)
            for team in lt.index:
                if lt.loc[team, ref_str] == '-' :
                    for opposing_team in lt.index:
                        if opposing_team != team and pgm.loc[team,opposing_team] and lt.loc[opposing_team, ref_str] == '-' :
                            lt.loc[team, ref_str] = opposing_team
                            lt.loc[opposing_team, ref_str] = team
                            pgm.loc[team, opposing_team] = False
                            pgm.loc[opposing_team, team] = False
                            nb_success +=2
                            if nb_success == lt.shape[0]:
                                keep_going = False
                            break     
        if verbose :
            print(f'Round number {round_number} - Unsuccessfull allocation based on rankings, random allocation was performed')                          
    return pgm, lt
