import numpy as np
import networkx as nx
import community

def Max_ner(lst,max_ner):
    for i in range(len(lst)):
        if len(lst[i]) >= max_ner:
            lst[i] = lst[i][:max_ner]
        else:
            length = len(lst[i])
            for _ in range(max_ner-length):
                lst[i].append(0)
    return lst

def get_community_member(partition,community_dict,node,kind):
    comm = community_dict[partition[node]]
    return [x for x in comm if x.startswith(kind)]

def prepare_vector_element(partition,relation,community_dict,num_user,num_item):

    item2user_neighbor_lst = [[] for _ in range(num_item)]     ###item的历史user
    user2item_neighbor_lst = [[] for _ in range(num_user)]     ###user的历史item
    flag_item = np.zeros(num_item)
    flag_user = np.zeros(num_user)

    for r in range(len(relation)):
        user, item = relation[r][0],relation[r][1]

        item2user_neighbor = get_community_member(partition, community_dict, user, 'u')
        np.random.shuffle(item2user_neighbor)

        user2item_neighbor = get_community_member(partition, community_dict, item, 'i')
        np.random.shuffle(user2item_neighbor)

        _,user = user.split('_',1)
        user = int(user)
        _,item = item.split('_',1)
        item = int(item)
        for i in range(len(item2user_neighbor)):
            name,index = item2user_neighbor[i].split('_', 1)
            item2user_neighbor[i] = int(index)
        for i in range(len(user2item_neighbor)):
            name,index = user2item_neighbor[i].split('_', 1)
            user2item_neighbor[i]=int(index)

        if flag_item[item]==0:
            item2user_neighbor_lst[item] = item2user_neighbor
            flag_item[item]=1
        if flag_user[user]==0:
            user2item_neighbor_lst[user] = user2item_neighbor
            flag_user[user]=1

    return user2item_neighbor_lst,item2user_neighbor_lst

def Louvain(pairs, neigh_sample_num, num_users, num_items, tuner_params):
    relation, tmp_relation = pairs, []
    for i in range(len(relation)):
        tmp_relation.append(['user_' + str(relation[i][0]), 'item_' + str(relation[i][1])])

    G = nx.Graph()
    G.add_edges_from(tmp_relation)
    resolution = tuner_params['resolution']
    partition = community.best_partition(G, resolution=resolution)

    community_dict = {}
    community_dict.setdefault(0, [])
    for i in range(len(partition.values())):
        community_dict[i] = []
    for node, part in partition.items():
        community_dict[part] = community_dict[part] + [node]

    tmp_user2item, tmp_item2user = prepare_vector_element(partition, tmp_relation, community_dict,
                                                          num_users, num_items)
    item_len, user_len = 111111, 111111
    item_max_len = 0
    for i in range(len(tmp_item2user)):
        if len(tmp_item2user[i]) < item_len:
            item_len = len(tmp_item2user[i])
        if len(tmp_item2user[i]) > item_max_len:
            item_max_len = len(tmp_item2user[i])
    for i in range(len(tmp_user2item)):
        if len(tmp_user2item[i]) < user_len:
            user_len = len(tmp_user2item[i])

    u_neigh = Max_ner(tmp_user2item, neigh_sample_num)
    i_neigh = Max_ner(tmp_item2user, neigh_sample_num)
    return u_neigh, i_neigh

def RandomNeigh(pairs, neigh_sample_num, num_users, num_items, tuner_params):
    ui_inters = np.zeros((num_users, num_items), dtype=np.int8)
    ui_inters[pairs[:, 0], pairs[:, 1]] = 1
    u_neigh, i_neigh = [], []
    for u in range(num_users):
        neigh_list = ui_inters[u].nonzero()[0]
        # 如果这个点没有邻居，就把邻居全置为0（之后需要更合理的修改）
        if len(neigh_list) == 0:
            u_neigh.append(neigh_sample_num * [0])
        else:
            mask = np.random.randint(0, len(neigh_list), (neigh_sample_num,))
            u_neigh.append(neigh_list[mask])
    for i in range(num_items):
        neigh_list = ui_inters[:, i].nonzero()[0]
        if len(neigh_list) == 0:
            i_neigh.append(neigh_sample_num * [0])
        else:
            mask = np.random.randint(0, len(neigh_list), (neigh_sample_num,))
            i_neigh.append(neigh_list[mask])
    return u_neigh, i_neigh