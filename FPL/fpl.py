import pandas as pd 
import random
from .initialisation.choose_team import *
from .initialisation.get_fixtures import *
from .observation_space.get_state import *
from .step_function.feature_engineering import *
from .interface import draw_pitch

class FPL():
    metadata = {'render.modes': ['human']}

    def __init__(self, player_info, fixture_data_finished, teams):
        '''
        Initialise the environment to run the Premier Fantasy Football
 

        Arguments:

        player_info {table} - Information on the players

        fixture_data_finished {table} - Fixture data for each week

        teams {table} - The teams in the league

        Returns
            None
        '''
        #super(CustomEnv, self).__init__()
        # Set up all tables
        self.player_info = player_info
        self.fixture_data_finished = fixture_data_finished
        self.teams = teams
        ## Filter out what players are actually worth being used
        self.player_pool, self.budget = choose_team(player_info)
        ## Get upcoming fixtures
        self.fixtures = get_fixtures(fixture_data_finished, teams)
        ## How many gameweeks are left
        self.gameweek = self.fixtures.iloc[0]['event']
        ## Current reward
        self.reward = 0
        ## Number of free transfers
        self.free_transfers = 1
        ## Transfer limit
        self.transfer_limit = 20
        ## How many transfers have been made
        self.transfers_made = 0
        ## Any penalties incurred
        self.penalties = 0
        ## current_player_state remains constant in this iteration
        self.weeks_remaining = len(self.fixtures['event'].unique())
        ## Is there remaning games ot play or is the league over
        self.episode_over = self.weeks_remaining < 1
        ## Players that can can't choose again
        self.taken_players = non_available_players(self.player_pool)
        ## Information on players that have been switched
        self.player_switch = {'old_player': None, 'old_player_team':None, 'old_player_position': None,
                             'new_player': None, 'new_player_team':None, 'budget': self.budget}
        ## Dictionary on current information in the league
        self.return_dict = {'players': self.player_pool, 'reward_accumulated': self.reward,
                      'game_week': self.gameweek, 'no_weeks_remaining': self.weeks_remaining,
                       'Penalty': 0, 'league_complete': self.episode_over}
        ## Generate features from the current team
        self.feature_generation = create_feature_stats(player_info, self.player_pool, self.fixtures, self.teams)
        ## Create statistics on all the current players
        self.player_stats = self.feature_generation.stats_table
        ## Create matrix of current players
        self.current_player_state, self.current_player_names = matrix_player_state(self.player_stats, self.player_pool)
        ## 
        game_dict = {**self.player_switch,**self.return_dict}
        self.summarylabels=['league_complete','game_week','budget','reward_accumulated','new_player','old_player']
        self.summary=pd.json_normalize(game_dict)[self.summarylabels]
        
    def play_match(self,player_data):
        '''
        Play a match

        This involves:
        - Updating the reward function
        - Removing the latest games played from the fixture table
        - Updating the return dictionary to reflect any chanegs that were made

        Arguments:

        player_data {dict} - Information on how players performed each week

        Returns
            None
        '''
        ## Update the reward for the current match that has been played
        self.reward += self._get_reward(self.player_pool, self.fixtures, player_data)
        ## Account for any penalties
        self.reward -= self.penalties
        ## Update the fixtures that are to be played in future matches
        self.fixtures = self.fixtures[self.fixtures['event'] != self.gameweek].sort_values(by =['event']) 
        ## Update the game week
        self.gameweek += 1
        ## Take away a week
        self.weeks_remaining -= 1
        # Check if any other games are to be played
        self.episode_over = self.weeks_remaining < 1
        ## Update the return dictionary
        self.return_dict = {'players': self.player_pool, 'reward_accumulated': self.reward,
                      'game_week': self.gameweek, 'transfers_made': self.transfers_made, 'no_fixtures_remaining': self.weeks_remaining,
                      'league_complete':self.episode_over}

    def _step(self, player_data, transfers, old_player_name = [], new_player_name = []):
        '''
        Take the next step to update the player pool

         This involves:
        - Deciding whether to transfer a player or not
        - Swapping out one player with another
        
        Arguments:
        player_data {dict} - Information on how players performed each week
        transfers {int} - Number of transfers
        old_player_name {list} - List of old player ids
        new_player_name {list} - List of new player ids

        Returns
            reward {int} - The new reward
        '''
        ## Update the statistics on each player
        self.player_stats = self.feature_generation._update_player_stats(player_data, self.gameweek, self.player_pool, self.fixtures) 
        ## if empty list, no transfers are needed 
        if transfers:

            ## Check if number of transfers are above the limit    
            if transfers > self.transfer_limit: 
                transfers = self.transfer_limit

            #only return if +ve to prevent -ve penalties
            extra_transfers = max(transfers-self.free_transfers,0)
            self.penalty = 4 * extra_transfers

            for i in range(len(old_player_name)):
                ## Increase the number of transfers made
                self.transfers_made += 1
                ## Take action and update
                self.player_pool = self._take_action(old_player_name[i], new_player_name[i])
                self.taken_players.append(self.new_player_name)
        
        ## Update the matrix of player statistics
        self.current_player_state, self.current_player_names = matrix_player_state(self.player_stats, self.player_pool)
        # Play the following weeks matches
        self.play_match(player_data)
       
        game_dict = {**self.player_switch,**self.return_dict}
        self.summary=self.summary.append(pd.json_normalize(game_dict))[self.summarylabels]
        return self.reward

    def _render(self):
        '''
        Display the current state of the league and the teams

        Arguments:

        Returns
            game_table {pandas} - Current status of the league
        '''
        game_dict = {**self.player_switch,**self.return_dict}
        return pd.json_normalize(game_dict)

    def _take_action(self, old_player_name, new_player_name):
        '''
        Take action and swap out old player for new player
        
        Arguments:
        old_player_name {int} - The id of the old player
        new_player_name {int}- The id of the new player
        player_stats {dict} - Statistcs on the players

        Returns
            team_dict {dict} - Updated team information
        '''

        # Set up the player info and current player pool        
        player_table = self.player_info
        team_dict = self.player_pool

        # Get position/name/team of new player
        new_player = player_table[player_table['name'] == new_player_name]
        self.new_player_name = new_player['name'].iloc[0]
        self.new_player_position = new_player['position'].iloc[0]
        self.new_player_team = new_player['team'].iloc[0]
        new_player_cost = list(player_table[player_table['name'] == self.new_player_name]['cost'])[0]
        
        # Get position/name/team of old player
        self.old_player = player_table[player_table['name'] == old_player_name]['name'].iloc[0]
        self.old_player_position = self.new_player_position
        self.old_player_team = list(player_table[player_table['name'] == self.old_player]['team'])[0]
        old_player_cost = list(player_table[player_table['name'] == self.old_player]['cost'])[0]
        
        ## Current price of new team
        self.budget = self.budget - new_player_cost + old_player_cost

        ## Swap in old player for new in the player pool dictionary
        team_dict[self.new_player_position] = [p for p in team_dict[self.new_player_position] if not p == self.old_player]
        team_dict[self.new_player_position].append(self.new_player_name)

        ## Get the player swicth information
        self.player_switch = {'old_player': self.old_player, 'old_player_team': self.old_player_team,
                              'old_player_position': self.old_player_position,
                             'new_player': self.new_player_name, 'new_player_team': self.new_player_team,
                             'budget': self.budget}
        
        return team_dict

    def _get_available_players(self):
        '''
        Get all the available players to choose from
        
        Arguments:
        

        Returns
            available_players {list} - List of available player ids
        '''

        player_table = self.player_info
        
        ## Cant take players currently on the team
        unavailable_players = self.current_player_names

        ## All players IDs that are available
        available_players = player_table[~player_table.name.isin(unavailable_players)]['Unnamed: 0'].values
        
        return available_players

    def _get_reward(self, player_pool, fixtures, player_data):
        '''
        Calculate the reward accumulated from each match
        
        Arguments:
        player_pool {dict} -  Dictionary of current players
        fixtures {pandas} - Table with fixture information for the season
        player_data {dict} -  Dictionary with information on each player

        Returns
            points {int} - The number of points accumlated on a given week
        '''
        points = 0
        gameweek = str(fixtures.iloc[0]['event'])
        positions = ['MID', 'DEF', 'FWD', 'GKP']
        for position in positions:
            for player in player_pool[position]:
                for game in player_data[player]:
                    if game['gameweek'] == gameweek: 
                        points += float(game['points'])
        return points

    def display_current_team(self):
        '''
        Display the current team 
        
        Arguments:
        

        Returns
            Displays the current team on a pitch
        '''
        draw_pitch.draw_player_pitch(self.player_pool)

    def _get_worst_performing_pool_player(self):

        '''
        Return the worst performing player in the player pool

        Arguments:

        Returns

            Returns the id of the worst performing player
        '''

        player_stats = self.player_stats
        # Get all availale players on the team
        swap_player_options = self.current_player_names
        # Filter out the player stats from the team
        filter_player_table = player_stats[player_stats.name.isin(swap_player_options)]
        ## Get the name of the player that is being swapped
        player =  filter_player_table[filter_player_table.avg_total_points == filter_player_table.avg_total_points.min()]['name'].values[0]

        return player

    def _get_best_available_performing_player(self, old_player_name, available_players):
        '''
        Return the best performing player from available players. It takes into account the position of the old player that is being removed. The "best" in this scenario
        is calauclated as the player who accumulated the most points of the league so far
        
        Arguments:
        old_player_name {str} - The name of the old player that is being removed
        available_players {list} - The list of players that are available to choose from 

        Returns
            Returns the name of the name of the best performing player given the above conditions
        '''
        player_stats = self.player_stats
        player_table = self.player_info
        
        ## Get position of player that is being swapped out
        position = player_table[player_table['name'] == old_player_name]['position'].values[0]
        ## Get the names of available players
        filter_player_table = player_table[player_table['Unnamed: 0'].isin(available_players)]
        ## Filter the table for available players
        available_player_names = list(filter_player_table[filter_player_table.position == position]['name'].values)
        ## Get statistics table of available players
        available_player_stats = player_stats[player_stats['name'].isin(available_player_names)]
        # Remove the player with the lowest accumulated points
        player =  available_player_stats[available_player_stats.avg_total_points == available_player_stats.avg_total_points.max()]['name'].values[0]
    
        return player