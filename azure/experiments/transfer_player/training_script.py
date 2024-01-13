import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import os, sys
sys.path.append('.')
from fpl import *
from itertools import count
from next_actions.agent import *
from next_actions.dqn import *
from azure.storage.blob import BlobServiceClient
import pandas as pd
import json
import mlflow
import torch

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

player_file = open('./data_2021/player_details_2021.json')
player_match_info = json.load(player_file)

player_match_info_refined = {k: player_match_info[k] for k in player_info_refined.name}

# Train the model 

## Set up the environment
env = FPL(player_info_refined_all, fixture_data_finished, teams)
agent = trainAgent(env.current_player_state)

## Increase the number of episodes to train
num_episodes = 10000
# control how lagged is target network by updating every n episodes
TARGET_UPDATE = 20

steps_done = 0
reward_list = []
## Current players is the number of players in player pool
current_players_ids = np.arange(15)
## Choose to skip, if greater index than all available player ids
skip_idx = 15
current_player_ids = np.append(current_players_ids, skip_idx)

# Run iteration 
for i in range(num_episodes):
    print("iteration: "+str(i))
    steps_done = 0
    # Reset board for each episode
    env = FPL(player_info_refined_all, fixture_data_finished, teams)
    ## Play the first match
    ## No swaps are needed at this point
    reward_p1 = env._step(player_match_info_refined, 0, [], [])
    
    # Keep going until someone wins or loses
    for t in count():
        # Get the current state of how every player is performing
        player_state = env.current_player_state.copy()
        current_player_names = env.current_player_names.copy()
        
        if t > 0:
            # Dump any reward player actions
            agent.memory.dump([old_state, action_player_id, reward_p1, player_state])
        
        ## Choose what action to take
        print("Steps: " + str(steps_done))
        action_player_name, action_player_id = agent.select_action(player_state, current_player_ids, current_player_names, steps_done)
        ## Check if a transfer is to be made
        if action_player_name:
            ## Allow only for one transfer at the moment
            transfers = 1
            ## Build function to update list week to week
            available_players = env._get_available_players()
            ## get the best player to replace them
            new_player_name = env._get_best_available_performing_player(action_player_name, available_players)
            
            ## Hard code only 1 valid transfer for now
            ## TODO - allow for more
            old_players = [action_player_name]
            new_players = [new_player_name]  

        else:
            transfers = 0 
            old_players = []
            new_players = []

        ## increment the steps
        steps_done += 1
        ## Update the reward

        reward_p1 = env._step(player_match_info_refined, transfers, old_players, new_players)
        
        ## Check if the league is over
        if env.episode_over:
            ## Dump the state - action - reward in memory
            agent.memory.dump([player_state, action_player_id, reward_p1, None])
            ## Add finishing reward to list
            #reward_list.append(reward_p1)
            mlflow.log_metric("reward", reward_p1)
            mlflow.log_metric("Transfers Made", env.transfers_made)
            #env.display_current_team()
            #plt.show()
            print('Running iteration {}, reward {}, transfers made {}'.format(i, reward_p1, env.transfers_made))
            break

        ## If environment is not over, the current state becomes the old state
        old_state = player_state
        
    # Perform one step of the optimization (on the policy network)
    agent.optimize_model()
        
    # update the target network, copying all weights and biases in DQN
    if i % TARGET_UPDATE == TARGET_UPDATE - 1:
        agent.update_target()        
        
        
print('Complete')
path = 'FPL_agent.pth'
mlflow.pytorch.log_model(agent.policy_net, path)