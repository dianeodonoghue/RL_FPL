import random

# memory block for deep q learning
class replayMemory:
    def __init__(self):
        # Set empty list for memoty
        self.memory = []
        
    def dump(self, transition_tuple):
        # Add a transition move
        # state, action, reward, next_state
        self.memory.append(transition_tuple)
    
    def sample(self, batch_size):
        # Take a random same from memory
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        # Check how many points are stored in memory
        return len(self.memory)
    
memory = replayMemory()