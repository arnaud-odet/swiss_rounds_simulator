import pandas as pd
import numpy as np
import string

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
    init_games = [True] * n_teams
    possible_games_matrix = pd.DataFrame([init_games] * n_teams, index = teams, columns = teams)
    for team in teams :
        possible_games_matrix.loc[team,team] = False
    
    calendar_columns = ['Level','Strategy','Nb_win','Nb_games',"Win_rate"]
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
    init_cal = [levels] + [[[]]*n_teams] + [[0]*n_teams] * 3 + ([['-']*n_teams] * (2*n_rounds))
    league_table = pd.DataFrame(init_cal, index= calendar_columns, columns = teams).transpose()
    
    for strat in strategies.keys():
        try :
            league_table.loc[strat,'Strategy'] = strategies[strat]
        except : 
            pass
    
    return possible_games_matrix , league_table.round(2)


def assign_opponents(possible_games_matrix, league_table, round_number, verbose = False):
    
    if f"R{round_number}_opponent" not in league_table.columns: 
        raise ValueError(f'The league was initiated with less than {round_number} games')
    
    ref_str = f'R{round_number}_opponent'
    
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
                        print('Successfull allocation based on ranking')
                    break
    

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
            print('Unsuccessfull allocation based on rankings, random allocation was performed')                          
    return pgm, lt


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


def play_round(league_table, round_number, method = 'probabilistic', verbose = False):
    
    if f"R{round_number}_opponent" not in league_table.columns: 
        raise ValueError(f'The league was initiated with less than {round_number} games')
    
    opp_str = f'R{round_number}_opponent'
    res_str = f'R{round_number}_result'
    for team in league_table.index:
        if league_table.loc[team, res_str] == '-':
            opponent = league_table.loc[team, opp_str]
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
                winner = team
            else :
                league_table.loc[opponent,'Nb_win'] +=1
                winner = opponent
            league_table.loc[team,'Nb_games'] +=1
            league_table.loc[opponent,'Nb_games'] +=1            
            if (team_level > opponent_level and result[0] == 'Loss') or team_level < opponent_level and result[0] == 'Win':
                unexpected = True
            else :
                unexpected = False
            
            if verbose :
                print(f'Round number {round_number}, game {team} vs {opponent} of levels {np.round(team_level,2)} vs {np.round(opponent_level,2)}: victory for {winner}{" - UNEXPECTED" if unexpected else ""} {" - STRATEGIC " if strat > 0 else ""}')

    league_table['Win_rate'] = league_table['Nb_win'] / league_table['Nb_games']
    
    return league_table.sort_values(by='Win_rate', ascending = False)  


def simulate_tournament(nb_teams, nb_games, strategies = {}, method = 'probabilistic', delta_level = 'linear', verbose=False):

    gdf, lt = initiate_league(nb_teams,nb_games, delta_level=delta_level,strategies=strategies)
    for i in range(nb_games):
        gdf, lt = assign_opponents(gdf,lt,i+1, verbose = verbose)
        lt = play_round(lt,i+1, method = method,verbose = verbose)

    return lt


def simulate_n_tournaments(n_tournaments, nb_teams, nb_games, strategies = {}, method = 'probabilistic', delta_level = 'linear') :

    first = True 
    
    for i in range(n_tournaments) :
        lt = simulate_tournament(nb_teams, nb_games, strategies = strategies, delta_level=delta_level,method = method)
        if first :
            out = lt[['Level','Strategy','Win_rate']].rename(columns={'Win_rate':'WR_0'})
            first = False
        else :
            temp_lt = lt[['Win_rate']].rename(columns={'Win_rate':f'WR_{i}'})
            out = out.merge(temp_lt, left_index=True, right_index=True)
    out['Avg_WR'] = out.drop(columns = ['Level','Strategy']).mean(axis=1)
    
    out = out[['Level','Strategy','Avg_WR']]
    out = out.round(2)
    return out.sort_values(by = 'Avg_WR', ascending = False)


def compare_settings(n_tournaments, n_teams, n_rounds, delta_level='linear' ,strategies={}, probabilistic=True, deterministic=True):
    
    if probabilistic :
        print('Probabilistic resolution')
        r = simulate_n_tournaments(n_tournaments,n_teams,n_rounds, delta_level=delta_level, method = 'probabilistic')
        r.rename(columns = {'Avg_WR': 'Control_avg_WR'}, inplace = True)
        rs = simulate_n_tournaments(n_tournaments,n_teams,n_rounds, delta_level=delta_level, method = 'probabilistic', strategies = strategies)
        rs.rename(columns = {'Avg_WR': 'Strategic_avg_WR'}, inplace = True)
        rs = rs.merge(r['Control_avg_WR'], left_index= True, right_index=True)
        rs['Delta'] = rs['Strategic_avg_WR'] - rs['Control_avg_WR']
        display(rs)
    
    if probabilistic and deterministic :
        print('---------------------------------------------------')
    
    if deterministic:
        print('Deterministic resolution')
        d = simulate_n_tournaments(n_tournaments,n_teams,n_rounds, delta_level=delta_level, method = 'deterministic')
        d.rename(columns = {'Avg_WR': 'Control_avg_WR'}, inplace = True)
        ds = simulate_n_tournaments(n_tournaments,n_teams,n_rounds, delta_level=delta_level, method = 'deterministic', strategies = strategies)
        ds.rename(columns = {'Avg_WR': 'Strategic_avg_WR'}, inplace = True)
        ds = ds.merge(d['Control_avg_WR'], left_index= True, right_index=True)
        ds['Delta'] = ds['Strategic_avg_WR'] - ds['Control_avg_WR']
        display(ds)  
        
    if not probabilistic and not deterministic :
        print('No setting were set to True for comparison')
