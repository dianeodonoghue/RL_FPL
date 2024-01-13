from mplsoccer import Pitch
import matplotlib.pyplot as plt


def draw_player_pitch(player_pool):
  '''
  Draw the pitch of the current players

  Inputs:
  player_pool: The pool of the current players that are being used

  returns:
  A matplotlib of the players
  '''
  pitch = Pitch(pitch_color="grass", line_color="white", stripe=True)
  fig, ax = pitch.draw(figsize=(10, 6))
  pos_x = {'MID': 50, 'DEF': 80, 'GKP': 100, 'FWD': 20}
  x_pos = []
  y_pos = []
  name_list = []

  for pos in player_pool.keys():
      
      len_players = len(player_pool[pos])
      y = 100 / len_players
      start = 100 / round((len_players + 2))
      ind = start
      for player in player_pool[pos]:
          name_list.append(player)
          y_pos.append(ind)
          x_pos.append(pos_x[pos])
          ind+=start
        

  sc = pitch.scatter(x_pos, y_pos, 
  #c=['red', 'blue', 'green', 'yellow', 'orange'],
        s=100, label='scatter', ax=ax)

  n = name_list # Labels for each point
  data = sc.get_offsets()

  #leg = ax.legend(borderpad=1, markerscale=0.5, labelspacing=1.5, loc='upper center', fontsize=15)

  for idx,label in enumerate(n):
      ax.annotate(label, (data[idx][0]+2, data[idx][1]))