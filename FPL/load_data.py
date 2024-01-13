import json
import csv
import pandas as pd

fixture_data_finished = pd.read_csv('./data_2021/fixture_data_2021.csv').sort_values(by = ['id'])
teams = pd.read_csv('./data_2021/teams_2021.csv')
player_info = pd.read_csv('./data_2021/player_info_2021.csv')
pre_player_info = pd.read_csv('./data_2021/previous_season_info.csv')

## Only choose from 70 best players
player_info_DEF_refined = player_info[player_info.position == 'DEF'].sort_values('cost', ascending = False)[0:30]
player_info_MID_refined = player_info[player_info.position == 'MID'].sort_values('cost', ascending = False)[0:30]
player_info_FWD_refined = player_info[player_info.position == 'FWD'].sort_values('cost', ascending = False)[0:20]
player_info_GKP_refined = player_info[player_info.position == 'GKP'].sort_values('cost', ascending = False)[0:10]

player_info_refined = player_info_DEF_refined.append(player_info_MID_refined).append(player_info_FWD_refined).append(player_info_GKP_refined)

player_info_refined['Unnamed: 0'] = range(len(player_info_refined))

pre_player_info = pre_player_info.drop('Unnamed: 0', axis = 1)

player_info_refined_all = pre_player_info.merge(player_info_refined, on = 'name', how = 'inner')

player_file = open('../data/player_details_2021.json')
player_match_info = json.load(player_file)

player_match_info_refined = {k: player_match_info[k] for k in player_info_refined.name}
