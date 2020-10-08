import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from neigh import Louvain, RandomNeigh
#from models.evaluation_metrics import sample_metric as smetric


class Dataset():
    def __init__(self, dataset_name, tuner_params):
        self.tuner_params = tuner_params
        np.random.seed(2020)
        neigh_sample_num = tuner_params['neigh_sample_num']
        self.neigh_sample_num = neigh_sample_num
        max_his_len = tuner_params['max_his_len']
        self.max_his_len = max_his_len
        phases = ['train', 'dev', 'test']
        data = {}
        for key in phases:
            data[key] = pd.read_csv(f'./data/{dataset_name}/{key}.csv', sep='\t') if key != 'train' \
                else np.genfromtxt(f'./data/{dataset_name}/{key}.csv', dtype=int, autostrip=True)
        data['train'] = data['train'][1:, :-1] - 1
        self.data = data
        self.num_users = max(data['train'][:, 0].max(), data['dev']['user_id'].max(), data['test']['user_id'].max()) + 1
        self.num_items = max(data['train'][:, 1].max(), data['dev']['item_id'].max(), data['test']['item_id'].max()) + 1
        self.feed_dict = {key: [] for key in phases}
        self.users_adj_list = {key: {} for key in phases}
        self.ui_inters = np.zeros((self.num_users, self.num_items), dtype=np.int8)
        self.ui_inters[data['train'][:, 0], data['train'][:, 1]] = 1

        self.neg_sample()

        for key in ['dev', 'test']:
            df = data[key]
            # Formating data type
            for col in df.columns:
                df[col] = df[col].apply(lambda x: eval(str(x)))
            u_ids = list(df['user_id'] - 1)
            i_ids = list(df['item_id'] - 1)
            neg_items = list(df['neg_items'].apply(lambda x: [i - 1 for i in x]))
            self.users_adj_list[key] = self.users_adj_list['train' if key == 'dev' else 'dev'].copy()
            for idx in range(len(u_ids)):
                if u_ids[idx] not in self.users_adj_list[key]:
                    self.users_adj_list[key][u_ids[idx]] = []
                row = {
                    'user': np.array(u_ids[idx]),
                    'item': np.array([i_ids[idx]] + neg_items[idx]),
                    'user_his': np.array(self.users_adj_list[key][u_ids[idx]][-max_his_len:]),
                    'user_len': np.array(len(self.users_adj_list[key][u_ids[idx]][-max_his_len:])),
                }
                self.users_adj_list[key][u_ids[idx]].append(i_ids[idx])
                self.feed_dict[key].append(row)
        neigh_method = Louvain if 'resolution' in tuner_params else RandomNeigh
        u_neigh, i_neigh = neigh_method(data['train'], self.neigh_sample_num, self.num_users, self.num_items,
                                        tuner_params)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.u_neigh = torch.tensor(u_neigh, dtype=torch.long, device=device)
        self.i_neigh = torch.tensor(i_neigh, dtype=torch.long, device=device)

    def neg_sample(self):
        train_nega_coef = self.tuner_params['neg_num']
        # idx:index对每一个trainPair产生1：5的negSample
        for idx, (u, i) in enumerate(self.data['train']):
            neg_items, cnt = [i], 0
            while cnt < train_nega_coef:
                items = np.random.randint(0, self.num_items, train_nega_coef - cnt)
                tlist = list(filter(lambda item: self.ui_inters[u, item] != 1, items))
                neg_items.extend(tlist)
                cnt += len(tlist)
            if len(self.feed_dict['train']) == idx:
                # 对每一个user创建很多TrainingPair
                if u not in self.users_adj_list['train']:
                    self.users_adj_list['train'][u] = []
                row = {
                    # 用户标号
                    'user': np.array(u),
                    # 这次购买对应的负样例
                    'item': np.array(neg_items),
                    # user_his代表购买记录，取最后20个
                    'user_his': np.array(self.users_adj_list['train'][u][-self.max_his_len:]),
                    # user_len表示实际的序列长度，一定小于等于20
                    'user_len': np.array(len(self.users_adj_list['train'][u][-self.max_his_len:])),
                }
                # user_adj_list记录user-item pair
                self.users_adj_list['train'][u].append(i)
                # feed_dict记录所有的信息。
                self.feed_dict['train'].append(row)
            else:
                self.feed_dict['train'][idx]['item'] = neg_items

def rechorus_loss(predictions):
    pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
    neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
    neg_pred = (neg_pred * neg_softmax).sum(dim=1)
    loss = F.softplus(-(pos_pred - neg_pred)).mean()
    return loss


def collate(feed_dicts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feed_dict = {}
    for key in feed_dicts[0]:
        stack_val = np.array([d[key] for d in feed_dicts])
        if stack_val.dtype == np.object:  # inconsistent length (e.g. history)
            feed_dict[key] = pad_sequence([torch.from_numpy(x).long().to(device=device) for x in stack_val],
                                          batch_first=True)
        else:
            feed_dict[key] = torch.from_numpy(stack_val).long().to(device=device)
    return feed_dict


def evaluate_method(predictions):
    predictions = predictions.cpu().data.numpy()
    topk = [5, 10]
    metrics = ['HR', 'NDCG']
    evaluations = dict()
    sort_idx = (-predictions).argsort(axis=1)
    gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
    # 防止模型的输出为：所有item具有相同分数，因为这会导致指标为1
    idx = (predictions[:, 0]==predictions[:, 1]).nonzero()[0]
    gt_rank[idx] = np.random.randint(1,101,idx.shape)
    for k in topk:
        hit = (gt_rank <= k)
        for metric in metrics:
            key = '{}@{}'.format(metric, k)
            if metric == 'HR':
                evaluations[key] = hit.mean()
            elif metric == 'NDCG':
                evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    return evaluations


def init_weights(m):
    if 'Linear' in str(type(m)):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
    elif 'Embedding' in str(type(m)):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
