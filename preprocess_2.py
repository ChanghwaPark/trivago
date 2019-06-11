import os.path

import numpy as np
import pandas as pd
from termcolor import colored

from utils import save_file, load_file

itemfile_name = 'dataset/item_metadata.csv'
trainfile_name = 'dataset/train.csv'
item_vector_path = 'files/item_vector.pkl'
user_id_dict_path = 'files/user_id_dict.pkl'

trainfile = load_file('trainfile')
train_size = trainfile.shape[0]
print("Train file is loaded.")
timestamp_dict = {}

for i in range(train_size):
    if i==0:
        time_start = trainfile[i, 2]

    if i < train_size -1:
        if trainfile[i, 1] != trainfile[(i+1), 1]:
            time_end = trainfile[i, 2]
            time_spent = time_end - time_start
            timestamp_dict.update({len(timestamp_dict): time_spent})
            time_start = trainfile[(i+1), 2]
    else:
        time_end = trainfile[i, 2]
        time_spent = time_end - time_start
        timestamp_dict.update({len(timestamp_dict): time_spent})

save_file(timestamp_dict, 'timestamp_dict')
# print(timestamp_dict[0])
# print(timestamp_dict[1])
# print(timestamp_dict[2])
# print(timestamp_dict[3])