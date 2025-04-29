# sampling.py

import numpy as np

def cbisddsm_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CBIS-DDSM dataset
    :param dataset: CBIS-DDSM Dataset
    :param num_users: Number of users
    :return: dict of image indices per user
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
