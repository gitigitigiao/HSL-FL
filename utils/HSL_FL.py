import math
import sys
import os
from os import path
sys.path.append('/home/workspace/python/HSL-FL' )
from utils.ucb1 import UCB1
import random
import math

class HSL_FL:
    def __init__(self,client_num,mu,dataset):
        self.H=dict()
        self.ability = dict()
        self.ability_count = dict()

        self.mu = mu

        self.ucb_ability = UCB1(list(),list(),client_num)
        self.ability_dict_name = dataset + "ability_dict"
        aaa = self.ucb_ability.load(self.ability_dict_name)
        self.user_list = list(range(client_num))

        self.ucb_comm = UCB1(list(),list(),client_num)
        bbb = self.ucb_comm.load("comm_dict")
        self.load_from_file = False
        if aaa and bbb:
            self.load_from_file = True

        self.ability_avg = 0.5
        self.roulette = []
        for i in range(client_num):
            self.roulette.append(self.ability_avg)

    def get_reward(self,user):
        r = self.mu*self.ucb_ability.predict_value(user) + (1-self.mu)*self.ucb_comm.predict_value(user)
        return r

    def get_ability_avg(self):
        return self.ability_avg
    def set_ability_avg(self,value):
        self.ability_avg = value

    def get_ucb_ability_predict(self,i):
        return self.ucb_ability.predict_value(i)
    def get_ucb_comm_predict(self,i):
        return self.ucb_comm.predict_value(i)
    def get_ucb_ability_value(self,i):
        return self.ucb_ability.get_value(i)
    def get_ucb_comm_value(self,i):
        return self.ucb_comm.get_value(i)
    def get_user_list(self):
        return self.user_list
    def shuffle_user_list(self):
        random.shuffle(self.user_list)

    def set_H_fixed(self,user, ability_pred, beta, T):
        self.H[user] = math.ceil(ability_pred * (beta + (1 - beta) * random.random()) * T)


    def save_ucb(self):
        self.ucb_ability.save(self.ability_dict_name)
        self.ucb_comm.save("comm_dict")