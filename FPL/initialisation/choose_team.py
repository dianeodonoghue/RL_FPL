def choose_team(player_table):
    ## for now choose random players
    ## dont care about budget
    ## Have a smarter way to do it and then optimise
        positions = {'MID':5, 'DEF':5, 'FWD':3, 'GKP':2}
        budget = 0
        team_dict = {}
        for position in positions.keys():
            players = player_table[player_table['position'] == position].sample(positions[position], random_state = 1234)
            team_dict[position] = list(players['name'])
            budget += players['cost'].sum()
        return team_dict, budget