import math
import random
import pickle

def ind_max(x):
  m = max(x)
  return x.index(m)

class UCB1():
  def __init__(self, counts, values, n_arms):
    self.counts = counts
    self.values = values
    self.n_arms = n_arms
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    self.flag = [1 for col in range(n_arms)]
    return
  
  def select_arm(self):
    n_arms = len(self.counts)
    arm_list = list(range(n_arms))
    random.shuffle(arm_list)
    for arm in arm_list:
      if self.counts[arm] == 0:
        return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
      ucb_values[arm] = self.values[arm] + bonus
    return ind_max(ucb_values)
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]

    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return
    
  def predict_value(self,arm):
    total_counts = sum(self.counts)
    bonus = math.sqrt((2.0 * math.log(float(total_counts))) / float(self.counts[arm]))
    ucb_values = self.values[arm] + bonus
    return ucb_values
  
  def get_value(self,arm):
    return self.values[arm]

  def save(self, name):
    obj = {}
    obj["counts"] = self.counts
    obj["values"] = self.values
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)

  def load(self,name="ucb_data"):
    import os
    if os.path.exists('./' + name + '.pkl') == True:
      with open('./' + name + '.pkl', 'rb') as f:
          obj = pickle.load(f)
          self.counts = obj["counts"]
          self.values = obj["values"]
          print("load ucb parm")
      return True
    else:
      self.initialize(self.n_arms)
      return False


