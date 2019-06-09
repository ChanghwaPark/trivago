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
    session_dict = {}

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
            current_session = trainfile[i, 1]

            if current_user_id not in user_id_dict:
                user_id_dict.update({current_user_id: len(user_id_dict)})
            if current_action_type not in action_type_dict:
                action_type_dict.update({current_action_type: len(action_type_dict)})
            if not current_reference_value.isdigit() and type(current_reference_value) == np.str:
                current_reference_values = current_reference_value.split(", ")
                for j in range(len(current_reference_values)):
                    if current_reference_values[j] not in reference_value_dict:
                        reference_value_dict.update({current_reference_values[j]: len(reference_value_dict)})
            if current_platform not in platform_dict:
                platform_dict.update({current_platform: len(platform_dict)})
            if type(current_city) == np.str:
                current_cities = current_city.split(", ")
                for j in range(len(current_cities)):
                    if current_cities[j] not in city_dict:
                        city_dict.update({current_cities[j]: len(city_dict)})
            if current_device not in device_dict:
                device_dict.update({current_device: len(device_dict)})
            if type(current_filter) == np.str:
                current_filters = current_filter.split("|")
                for j in range(len(current_filters)):
                    if current_filters[j] not in filter_dict:
                        filter_dict.update({current_filters[j]: len(filter_dict)})
            if current_session not in session_dict:
                session_dict.update({current_session: len(session_dict)})

    print(f"User id: {len(user_id_dict)}")
    print(f"Action type: {len(action_type_dict)}")
    print(f"Reference value: {len(reference_value_dict)}")
    print(f"Platform: {len(platform_dict)}")
    print(f"City: {len(city_dict)}")
    print(f"Device: {len(device_dict)}")
    print(f"Filter: {len(filter_dict)}")
    print(f"Session: {len(session_dict)}")

    save_file(user_id_dict, 'user_id_dict')
    save_file(action_type_dict, 'action_type_dict')
    save_file(reference_value_dict, 'reference_value_dict')
    save_file(platform_dict, 'platform_dict')
    save_file(city_dict, 'city_dict')
    save_file(device_dict, 'device_dict')
    save_file(filter_dict, 'filter_dict')
    save_file(session_dict, 'session_dict')


def count_max_actions(trainfile, train_size):
    previous_session_id = ''
    count = 0
    max_count = 0
    max_actions_list = []

    for i in range(train_size):
        if previous_session_id != trainfile[i, 1]:
            if max_count < count:
                max_count = count
                max_actions_list.append(max_count)
            count = 1
            previous_session_id = trainfile[i, 1]

        else:
            if not (trainfile[(i - 1), 4:9] == trainfile[i, 4:9]).all():
                count += 1
            previous_session_id = trainfile[i, 1]

    save_file(max_actions_list, 'max_actions')
    print(f"Max actions in one session is {max_count}")


def make_session_vector():
    print(colored("Making session vector is started.", "blue"))
    item_vector = load_file('item_vector')
    user_id_dict = load_file('user_id_dict')
    action_type_dict = load_file('action_type_dict')
    reference_value_dict = load_file('reference_value_dict')
    platform_dict = load_file('platform_dict')
    city_dict = load_file('city_dict')
    device_dict = load_file('device_dict')
    filter_dict = load_file('filter_dict')
    session_dict = load_file('session_dict')
    print("Dict files are loaded.")

    user_id_len = len(user_id_dict)
    action_type_len = len(action_type_dict)
    # reference_value_len = len(reference_value_dict)
    reference_value_len = len(item_vector[5101])
    platform_len = len(platform_dict)
    city_len = len(city_dict)
    device_len = len(device_dict)
    filter_len = len(filter_dict)
    session_len = len(session_dict)

    user_id_space = user_id_len
    platform_space = platform_len
    city_space = city_len
    device_space = device_len

    trainfile = load_file('trainfile')
    train_size = trainfile.shape[0]
    print("Train file is loaded.")

    max_actions_path = 'files/max_actions.pkl'
    if not os.path.isfile(max_actions_path):
        count_max_actions(trainfile, train_size)
    else:
        print(colored("Max actions file already exists. Skipped the saving process", "blue"))

    max_actions_list = load_file('max_actions')
    # max_count = max(max_actions_list)
    # max_count = max_actions_list[len(max_actions_list) - 3]
    max_count = max_actions_list[len(max_actions_list) - 1]
    action_type_space = max_count * action_type_len
    # reference_value_space = max_count * (len(reference_value_dict) + len(item_vector[5101]))
    # reference_value_space = max_count * len(item_vector[5101])
    reference_value_space = max_count * reference_value_len
    filter_space = max_count * filter_len
    # impressions_space = 25 * len(item_vector[5101])
    impressions_space = 25 * reference_value_len
    prices_space = 25
    # total_space = user_id_space + action_type_space + reference_value_space + platform_space + city_space + device_space + filter_space
    total_space = action_type_space + reference_value_space + platform_space + city_space + device_space + filter_space + impressions_space + prices_space

    # print(user_id_space)
    print(action_type_space)
    print(reference_value_space)
    print(platform_space)
    print(city_space)
    print(device_space)
    print(filter_space)
    print(impressions_space)
    print(prices_space)
    print(colored(f"Total space for each session is {total_space}.", "blue"))
    print(f"train_size: {train_size}")
    print(f"session_size: {session_len}")

    impressions_dict = {}
    train_index = 0
    platform_start = 0
    city_start = platform_start + platform_space
    device_start = city_start + city_space
    action_type_start = device_start + device_space
    reference_value_start = action_type_start + action_type_space
    filter_start = reference_value_start + reference_value_space
    impressions_start = filter_start + filter_space
    prices_start = impressions_start + impressions_space
    # print(prices_start)
    # pcd_space = platform_space + city_space + device_space
    # max_action_index = 0

    for k in range(1000):
        session_file_name = "session_vector_{}".format(k)
        if k % 100 == 0:
            print(session_file_name)
        session_vector = np.zeros((int(session_len / 1000), total_space))
        # previous_session_id = ''
        session_index = 0
        action_index = 0
        # Indicator for platform, city and device update
        pcd_updated = False

        while session_index < int(session_len / 1000):

            # if session_index % 10000 == 0 and not pcd_updated:
            #     print(session_index)

            # Update action_index'th action in session_index'th session vector
            # if action_index == 0:
            if not pcd_updated:
                # Update platform information
                if trainfile[train_index, 6] in platform_dict:
                    session_vector[session_index, platform_dict[trainfile[train_index, 6]]] = 1

                # Update city information
                current_city = trainfile[train_index, 7]
                current_cities = current_city.split(", ")
                for j in range(len(current_cities)):
                    if current_cities[j] in city_dict:
                        session_vector[session_index, (city_start + city_dict[current_cities[j]])] = 1

                # Update device information
                if trainfile[train_index, 8] in device_dict:
                    session_vector[
                        session_index, (device_start + device_dict[trainfile[train_index, 8]])] = 1

                pcd_updated = True

            # previous_session_id = trainfile[train_index, 1]
            # train_index += 1

            # if not (trainfile[(train_index - 1), 4:9] == trainfile[train_index, 4:9]).all():
            if not (trainfile[train_index, 4:9] == trainfile[(train_index + 1), 4:9]).all():
                # Update action, reference and filter
                if trainfile[train_index, 4] in action_type_dict:
                    if trainfile[train_index, 4] != 'clickout item':
                        session_vector[session_index, (
                                action_type_start + action_index * action_type_len + action_type_dict[
                            trainfile[train_index, 4]])] = 1
                    if trainfile[train_index, 4] == 'clickout item':
                        current_impression = trainfile[train_index, 10]
                        current_impressions = current_impression.split("|")
                        current_price = trainfile[train_index, 11]
                        current_prices = current_price.split("|")
                        impressions_dict.update({trainfile[train_index, 1]: current_impressions})

                        for l in range(len(current_impressions)):
                            if (impressions_start + (l + 1) * reference_value_len) > prices_start:
                                raise ValueError("Impression value indexing error")
                            if int(current_impressions[l]) in item_vector:
                                session_vector[session_index, (impressions_start + l * reference_value_len):(
                                        impressions_start + (l + 1) * reference_value_len)] = item_vector[
                                    int(current_impressions[l])]

                        for l in range(len(current_prices)):
                            session_vector[session_index, (prices_start + l)] = current_prices[l]

                if trainfile[train_index, 5].isdigit() and int(trainfile[train_index, 5]) in item_vector:
                    if (reference_value_start + (action_index + 1) * reference_value_len) > filter_start:
                        raise ValueError("Reference value indexing error")
                    session_vector[session_index, (reference_value_start + action_index * reference_value_len):(
                            reference_value_start + (action_index + 1) * reference_value_len)] = item_vector[
                        int(trainfile[train_index, 5])]

                current_filter = trainfile[train_index, 9]
                if type(current_filter) == np.str:
                    current_filters = current_filter.split("|")
                    for j in range(len(current_filters)):
                        if current_filters[j] in filter_dict:
                            session_vector[session_index, (
                                    filter_start + action_index * filter_len + filter_dict[current_filters[j]])] = 1

                action_index += 1

            # if trainfile[(train_index - 1), 1] != trainfile[train_index, 1]:
            if trainfile[train_index, 1] != trainfile[(train_index + 1), 1]:
                session_index += 1

                # if max_action_index < action_index:
                #     max_action_index = action_index
                #     print(max_action_index)

                action_index = 0
                pcd_updated = False

            train_index += 1

        # print(session_index)
        # print(train_index)
        save_file(session_vector, session_file_name)
        # print(session_vector[0, pcd_space:(pcd_space + max_count * action_type_len)])
        # print(session_vector[1, 0:platform_space])
        # print(session_vector[2, 0:platform_space])
        # print(session_vector[3, 0:platform_space])
        # print(session_vector[4, 0:platform_space])
        # print(session_vector[0])

    # impressions_dict only contains item_ids
    save_file(impressions_dict, 'impressions_dict')


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

# Make session vectors for 1A
make_session_vector()
