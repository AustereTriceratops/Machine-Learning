import numpy as np
from keras.layers import *

# TODO:
# Logging simulation data to render externally
# 

class Agent():
  def __init__(self):
    self.x = 0 
    self.y = 0
    self.mind = None  # f: environmental information -> action policy
    self.history = [(self.x, self.y)]
    self.score_history = [1.0]  # change one of these for consistency
    self.score = 1.0

  def move(self, time_step=0.1):
    # TODO: update based on policy
    self.x += 0.8 * time_step
    self.y += 0.8 * time_step
    self.score -= 0.2*time_step
    self.score_history.append(self.score)
    self.history.append((self.x, self.y))
    

class Food():
  def __init__(self):
    self.x = np.random.uniform(high=2.0)
    self.y = np.random.uniform(high=2.0)
    self.benefit = 1 # (-)hazard vs (+) help
    

class Environment():
  def __init__(self):
    self.agent = Agent()
    self.food = [Food() for _ in range(8)]
    self.time = 0

  def simulate(self, steps):
    agent = self.agent
    for _ in range(steps):
      # give agent all data about the environment that it needs
      self.inform(agent)
      # agent's mind takes this information and adopts a policy
      # agent acts according to policy

      agent.move()

      for f in self.food:     # checks for a collision event i.e. food, threat
        if collision_event(agent, f):
          agent.score += f.benefit
          f.__init__()        # food respawns somewhere else

      self.time += 1

  def inform(self, agent_): #TODO: implement
    pass
    

def distance_from(agent_, object_):
  r = np.sqrt((agent_.x - object_.x)**2 + (agent_.y - object_.y)**2)
  return r


def collision_event(agent_, food_):
  if np.abs(agent_.x - food_.x) > 0.2 or np.abs(agent_.y - food_.y) > 0.2:
    return False
  else:
    return distance_from(agent_, food_) <= 0.1
