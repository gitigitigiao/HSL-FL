import sys
sys.path.append('/home/workspace/python/HSL-FL' )
import time
import math

def sleep_user(user):
    t = time.time()
    if user>=0 and user<30:
        sleep_time = math.ceil(10)
    elif user>=30 and user<70:
        sleep_time = math.ceil(3)
    else:
        sleep_time = 0
    return sleep_time

def get_comm_list(user_number):
    shuffle_user = [4, 39, 79, 88, 69, 8, 95, 70, 5, 68, 35, 2, 94, 80, 66, 96, 67, 52, 89, 86, 0, 87, 13, 20, 9, 11, 34, 16, 51, 85, 18, 50, 47, 61, 15, 57, 17, 91, 71, 1, 31, 53, 73, 30, 75, 19, 55, 12, 77, 28, 42, 32, 3, 76, 23, 44, 90, 81, 26, 56, 25, 41, 98, 33, 58, 10, 6, 97, 49, 54, 60, 62, 38, 63, 24, 37, 21, 82, 14, 36, 29, 22, 65, 99, 83, 46, 74, 84, 64, 93, 72, 48, 59, 27, 45, 92, 43, 7, 40, 78]
    comm_list = list()
    for i in range(user_number):
        comm_list.append(0)
    
    j=0
    for i in shuffle_user:
        if j<user_number*0.3:
            comm_list[i] = 25
        elif j<user_number*0.7:
            comm_list[i] = 50
        else:
            comm_list[i] = 100
        j = j+1

    return comm_list