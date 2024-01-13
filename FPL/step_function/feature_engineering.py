import pandas as pd

class create_feature_stats():

    def __init__(self, player_info, player_pool, fixtures, teams):

        self.stats_keys = ["points", "goals", "assists", "conceded", "bonus", "minutes", "clean_sheets", "own_goals", 
                    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bps", 
                    "influence", "creativity", "threat", "ict_index", "transfers_balance", "transfers_in", 
                    "transfers_out"]
        
        self.last_season_keys = ['no_games', 'avg_assists', 'last_assists', 'avg_bonus',
                    'last_bonus', 'avg_bps', 'last_bps', 'avg_clean_sheets',
                    'last_clean_sheets', 'avg_creativity', 'last_creativity',
                    'avg_element_code', 'last_element_code', 'avg_end_cost',
                    'last_end_cost', 'avg_goals_conceded', 'last_goals_conceded',
                    'avg_goals_scored', 'last_goals_scored', 'avg_ict_index',
                    'last_ict_index', 'avg_influence', 'last_influence', 'avg_minutes',
                    'last_minutes', 'avg_own_goals', 'last_own_goals',
                    'avg_penalties_missed', 'last_penalties_missed', 'avg_penalties_saved',
                    'last_penalties_saved', 'avg_red_cards', 'last_red_cards', 'avg_saves',
                    'last_saves', 'avg_start_cost', 'last_start_cost', 'avg_threat',
                    'last_threat', 'avg_total_points', 'last_total_points',
                    'avg_yellow_cards', 'last_yellow_cards']

        self.positions = ['MID', 'DEF', 'FWD', 'GKP']
        self.stats_extra = ['last_match_' + keys for keys in self.stats_keys]

        self.stats_dict = {k: [] for k in ['name'] + self.stats_keys +  self.stats_extra + self.positions + self.last_season_keys +["games_played", "prev_played" , "curr_team" , "no_matches_next_week" , "no_matches_remaining", "cost"]}
        players = player_info['name']

        ## update fixtures games remaining 
        self.all_teams = list(teams['name'].unique())
        self.fixture_week_next , self.fixture_week_remaining = self._update_games_remaining_teams_fixtures(fixtures, 0)

        for player in players:
            self.stats_dict['name'].append(player)
            
            for key in self.stats_keys +  self.stats_extra + ["games_played", "prev_played"]:
           
                self.stats_dict[key].append(0)
            player_row = player_info[player_info['name'] == player]
            player_position = player_row['position'].iloc[0]

            for pos in self.positions:
                if player_position == pos:
                    self.stats_dict[pos].append(1)
                else:
                    self.stats_dict[pos].append(0)

            if player in player_pool[pos]:
                self.stats_dict['curr_team'].append(1)
            else:
                self.stats_dict['curr_team'].append(0)

            ## update remaining matches
            player_team = player_row['team'].iloc[0]
            self.stats_dict["no_matches_next_week"].append(self.fixture_week_next[player_team])
            self.stats_dict["no_matches_remaining"].append(self.fixture_week_remaining[player_team])

            ## Update pre season_info
            self.stats_dict['cost'] = player_row['cost'].iloc[0]
            

            for key_info in self.last_season_keys:
                self.stats_dict[key_info].append(player_row[key_info].iloc[0])

        self.stats_table = pd.DataFrame.from_dict(self.stats_dict)

        
    def _update_player_stats(self, player_data, gameweek, player_pool, fixtures):
        '''
        Update the player statistics as the league goes on
        '''
        players = self.stats_table['name']
        for player in players:
            for game in player_data[player]:
                if game['gameweek'] == str(gameweek):
                    self.stats_table.loc[self.stats_table['name'] == player, 'games_played'] +=1 
                    for key in self.stats_keys:
                        self.stats_table.loc[self.stats_table['name'] == player, key] = self.stats_table.loc[self.stats_table['name'] == player, key] + float(game[key])
                        self.stats_table.loc[self.stats_table['name'] == player, 'last_'+key] = float(game[key])

                ## If player is being currently played
            player_pos = player_data[player][0]['position']
            if player_pos == 'GK':
                player_pos = 'GKP'
            if player in player_pool[player_pos]:
                self.stats_table.loc[self.stats_table['name'] == player, 'curr_team'] = 1
            ## Check if they were previously on a team
            elif self.stats_table.loc[self.stats_table['name'] == player, 'curr_team'].values[0] == 1:
                self.stats_table.loc[self.stats_table['name'] == player, 'curr_team'] = 0
                self.stats_table.loc[self.stats_table['name'] == player, 'prev_played'] = 1
            ## Update next week matches and remaining matches
            curr_team = player_data[player][0]['played_for']
            self.fixture_week_next , self.fixture_week_remaining = self._update_games_remaining_teams_fixtures(fixtures, gameweek)
            self.stats_table.loc[self.stats_table['name'] == player, 'no_matches_next_week'] = self.fixture_week_next[curr_team]
            self.stats_table.loc[self.stats_table['name'] == player, 'no_matches_remaining'] = self.fixture_week_remaining[curr_team]

        return self.stats_table

    def _update_games_remaining_teams_fixtures(self, fixtures, gameweek):
        team_match_dict_remaining = {}
        next_team_match_dict = {}
        next_match = gameweek + 1
        fixture_week_next = fixtures[fixtures['event'] == next_match]
        fixture_week_remaining = fixtures[fixtures['event'] > gameweek]

        for team in self.all_teams:
            team_match_dict_remaining[team] = len(fixture_week_remaining[(fixture_week_remaining['name_away'] == team) | (fixture_week_remaining['name_home'] == team)])
            next_team_match_dict[team] = len(fixture_week_next[(fixture_week_next['name_away'] == team) | (fixture_week_next['name_home'] == team)])

        return next_team_match_dict , team_match_dict_remaining


        


        

