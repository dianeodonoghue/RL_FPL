import pandas as pd 

def get_fixtures(fixture_data_finished, teams):
        merge_team_info = pd.merge(
            pd.merge(fixture_data_finished, teams, how = 'inner', left_on = 'team_a', right_on = 'team_id'),
            teams, how = 'inner', left_on = 'team_h', right_on = 'team_id', suffixes = ('_away', '_home'))

        return merge_team_info[['event', 'id', 'team_id_away', 'team_id_home', 'name_away', 'name_home']].sort_values(by =['event'])