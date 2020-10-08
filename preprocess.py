import numpy as np
import pandas as pd
import os

np.random.seed(2020)
# dataset_path = 'data/ml-1m'
# pairs = np.genfromtxt(os.path.join('data/ml-1m', 'ratings.dat'), dtype=int, delimiter='::', autostrip=
#     True, usecols=[0, 1, 3])
# movieLens Row=[user|item|rating|timeStamp]
dataset_path = 'data/ml-100k'
pairs = np.genfromtxt(os.path.join(dataset_path, 'u.data'), dtype=int,
                      autostrip=True, usecols=[0, 1, 3])

# 按时间戳排序
pairs = pairs[np.argsort(pairs[:, 2])]

# 得到item总个数
num_users = pairs[:, 0].max() + 1
num_items = pairs[:, 1].max() + 1

users_adj_list = [[] for _ in range(num_users)]
items_adj_list = [[] for _ in range(num_items)]
ui_t = np.zeros((num_users, num_items))

for (u, i, t) in pairs:
    users_adj_list[u].append(i)
    items_adj_list[i].append(u)
    ui_t[u, i] = t
# 生成训练/测试/开发？数据集
posi = {i: [] for i in ['train', 'dev', 'test']}
for i in range(num_users):
    # user对应的relativeItem数
    num_inters = len(users_adj_list[i])
    # 0.8之后的是校验集，0.9以后是测试集，（8，1，1）的比例
    val_begin = int(num_inters * 0.8)
    test_begin = int(num_inters * 0.9)
    for j in users_adj_list[i][:val_begin]:
        posi['train'].append([i, j, ui_t[i, j]])
    for j in users_adj_list[i][val_begin:test_begin]:
        posi['dev'].append([i, j, ui_t[i, j]])
    for j in users_adj_list[i][test_begin:]:
        posi['test'].append([i, j, ui_t[i, j]])
posi = {key: np.array(posi[key], dtype=np.int) for key in posi}

df = pd.DataFrame(data=posi['train'], columns=['user_id', 'item_id', 'time'])
df.to_csv(os.path.join(dataset_path, 'train.csv'), sep='\t', index=False)

df = {}
ui_inters = np.zeros((num_users, num_items), dtype=np.int8)
# 如果(u,i)在train中存在，则ui_inters置1
ui_inters[posi['train'][:, 0], posi['train'][:, 1]] = 1
tmp = ['dev', 'test']
for key in tmp:
    df[key] = {
        'user_id': [],
        'item_id': [],
        'time': [],
        'neg_items': []
    }
    ui_inters[posi[key][:, 0], posi[key][:, 1]] = 1
    for u, i, t in posi[key]:
        df[key]['user_id'].append(u)
        df[key]['item_id'].append(i)
        df[key]['time'].append(t)
        # 生成negtiveItem:策略是sample出index,然后判断是否是relativeItem,不是则加入list。
        row, cnt = [], 0
        while cnt < 99:
            items = np.random.randint(1, num_items, 99 - cnt)
            tlist = list(filter(lambda item: ui_inters[u, item] != 1, items))
            row.extend(tlist)
            cnt += len(tlist)
        df[key]['neg_items'].append(row)
    df[key] = pd.DataFrame.from_dict(df[key])
    df[key].to_csv(os.path.join(dataset_path, f'{key}.csv'), sep='\t', index=False)
