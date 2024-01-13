def matrix_player_state(player_info, team_dict):
    ## Features should be created here that will then be passed to the DQN

    list_player_options = list(team_dict.values())
    swap_player_options = [item for sublist in list_player_options for item in sublist]
    player_refined = player_info[player_info.name.isin(swap_player_options)]
    names = player_refined['name'].values
    matrix = player_refined.drop(columns = ['name']).values

    return matrix, names

def non_available_players(player_pool):
    player_list = []
    for position in player_pool.keys():
        player_list +=  player_pool[position]

    return player_list