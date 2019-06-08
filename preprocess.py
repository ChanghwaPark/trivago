import math
import os.path

import numpy as np
import pandas as pd
from termcolor import colored

from utils import save_file, load_file

itemfile_name = 'dataset/item_metadata.csv'
trainfile_name = 'dataset/train.csv'
item_vector_path = 'files/item_vector.pkl'
user_id_dict_path = 'files/user_id_dict.pkl'


def save_item_vector():
    itemfile = pd.read_csv(itemfile_name)
    itemfile = np.array(itemfile)
    item_size = itemfile.shape[0]
    item_features = []

    for i in range(item_size):
        features = itemfile[i, 1].split("|")
        for j in range(len(features)):
            if features[j] not in item_features:
                item_features.append(features[j])

    feature_size = len(item_features)
    item_vector = {}
    for i in range(item_size):
        vector = np.zeros((feature_size,))
        features = itemfile[i, 1].split("|")
        for j in range(len(features)):
            feature_index = item_features.index(features[j])
            vector[feature_index] = 1.
        item_vector.update({itemfile[i, 0]: vector})
    save_file(item_vector, 'item_vector')


def save_session_dicts():
    user_id_dict = {}
    action_type_dict = {}
    reference_value_dict = {}
    platform_dict = {}
    city_dict = {}
    device_dict = {}
    filter_dict = {}

    all_train_size = 0

    trainfile_chunk = pd.read_csv(trainfile_name, chunksize=1000000)

    for trainfile in trainfile_chunk:
        trainfile = np.array(trainfile)
        train_size = trainfile.shape[0]

        all_train_size += train_size
        print(f"Processing {all_train_size}")

        for i in range(train_size):
            current_user_id = trainfile[i, 0]
            current_action_type = trainfile[i, 4]
            current_reference_value = trainfile[i, 5]
            current_platform = trainfile[i, 6]
            current_city = trainfile[i, 7]
            current_device = trainfile[i, 8]
            current_filter = trainfile[i, 9]

            if current_user_id not in user_id_dict:
                user_id_dict.update({current_user_id:len(user_id_dict)})
            if current_action_type not in action_type_dict:
                action_type_dict.update({current_action_type:len(action_type_dict)})
            if not current_reference_value.isdigit() and type(current_reference_value) == np.str:
                current_reference_values = current_reference_value.split(", ")
                for j in range(len(current_reference_values)):
                    if current_reference_values[j] not in reference_value_dict:
                        reference_value_dict.update({current_reference_values[j]:len(reference_value_dict)})
            if current_platform not in platform_dict:
                platform_dict.update({current_platform:len(platform_dict)})
            if type(current_city) == np.str:
                current_cities = current_city.split(", ")
                for j in range(len(current_cities)):
                    if current_cities[j] not in city_dict:
                        city_dict.update({current_cities[j]:len(city_dict)})
            if current_device not in device_dict:
                device_dict.update({current_device:len(device_dict)})
            if type(current_filter) == np.str:
                current_filters = current_filter.split("|")
                for j in range(len(current_filters)):
                    if current_filters[j] not in filter_dict:
                        filter_dict.update({current_filters[j]:len(filter_dict)})

    print(f"User id: {len(user_id_dict)}")
    print(f"Action type: {len(action_type_dict)}")
    print(f"Reference value: {len(reference_value_dict)}")
    print(f"Platform: {len(platform_dict)}")
    print(f"City: {len(city_dict)}")
    print(f"Device: {len(device_dict)}")
    print(f"Filter: {len(filter_dict)}")

    save_file(user_id_dict, 'user_id_dict')
    save_file(action_type_dict, 'action_type_dict')
    save_file(reference_value_dict, 'reference_value_dict')
    save_file(platform_dict, 'platform_dict')
    save_file(city_dict, 'city_dict')
    save_file(device_dict, 'device_dict')
    save_file(filter_dict, 'filter_dict')

# Make the item vector
if not os.path.isfile(item_vector_path):
    save_item_vector()
else:
    print(colored("Item vector already exists. Skipped the saving process", "blue"))

# Make session dictionaries
if not os.path.isfile(user_id_dict_path):
    save_session_dicts()
else:
    print(colored("Session dicts already exist. Skipped the saving process", "blue"))
