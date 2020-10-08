import torch
import torch.nn as nn

class NNCF(nn.Module):
    def __init__(self, dataset, tuner_params):
        super(NNCF, self).__init__()
        conv_kernel_size = tuner_params['conv_kernel_size']
        pool_kernel_size = tuner_params['pool_kernel_size']
        conv_out_channels = tuner_params['conv_out_channels']

        self.user_embedding = nn.Embedding(dataset.num_users, tuner_params['embed_size'])
        self.items_enmbeding = nn.Embedding(dataset.num_items, tuner_params['embed_size'])

        self.user_neigh_embed = nn.Embedding(dataset.num_items, tuner_params['embed_size'])
        self.items_neigh_embed = nn.Embedding(dataset.num_users, tuner_params['embed_size'])

        self.users_conv = nn.Sequential(
            nn.Conv1d(tuner_params['embed_size'], conv_out_channels, kernel_size=conv_kernel_size),
            nn.MaxPool1d(pool_kernel_size),
            nn.ReLU()
        )
        self.items_conv = nn.Sequential(
            nn.Conv1d(tuner_params['embed_size'], conv_out_channels, kernel_size=conv_kernel_size),
            nn.MaxPool1d(pool_kernel_size),
            nn.ReLU()
        )
        conved_size = dataset.neigh_sample_num - (conv_kernel_size - 1)
        pooled_size = (conved_size - (pool_kernel_size - 1) - 1) // pool_kernel_size + 1

        hidden_num = [2 * pooled_size * conv_out_channels + tuner_params['embed_size'], tuner_params['hidden_size']]
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_num[i - 1], hidden_num[i])
                                                  for i in range(1, len(hidden_num))])
        self.out_layer = torch.nn.Linear(hidden_num[-1], 1, bias=False)
        self.dropout_layer = torch.nn.Dropout(p=tuner_params['dropout'])
        self.act = torch.nn.ReLU()

    def forward(self, feed_dict, dataset):
        user = feed_dict['user']
        items = feed_dict['item']
        user = user.unsqueeze(-1).repeat((1, items.shape[1]))
        user_embed = self.user_embedding(user)
        items_embed = self.items_enmbeding(items)

        user_neigh_input = dataset.u_neigh[user]
        user_neigh_emb = self.user_neigh_embed(user_neigh_input)
        batch_size = user_neigh_emb.size(0)
        dim = user_neigh_emb.size(1)
        user_neigh_emb = user_neigh_emb.view(batch_size * dim, user_neigh_emb.size(2), user_neigh_emb.size(3))
        user_neigh_emb = user_neigh_emb.permute(0, 2, 1)
        user_neigh_emb_conv = self.users_conv(user_neigh_emb)
        user_neigh_emb_conv = user_neigh_emb_conv.view(user_neigh_emb_conv.size(0), -1)
        user_neigh_emb_conv = user_neigh_emb_conv.view(batch_size, dim, -1)
        user_neigh_emb_conv = self.dropout_layer(user_neigh_emb_conv)

        items_neigh_input = dataset.i_neigh[items]
        items_neigh_emb = self.items_neigh_embed(items_neigh_input)
        items_neigh_emb = items_neigh_emb.view(batch_size * dim, items_neigh_emb.size(2), items_neigh_emb.size(3))
        items_neigh_emb = items_neigh_emb.permute(0, 2, 1)
        items_neigh_emb_conv = self.items_conv(items_neigh_emb)
        items_neigh_emb_conv = items_neigh_emb_conv.view(items_neigh_emb_conv.size(0), -1)
        items_neigh_emb_conv = items_neigh_emb_conv.view(batch_size, dim, -1)
        items_neigh_emb_conv = self.dropout_layer(items_neigh_emb_conv)

        mf_vec = user_embed * items_embed
        last = torch.cat((mf_vec, user_neigh_emb_conv, items_neigh_emb_conv), dim=-1)

        for layer in self.hidden_layers:
            last = layer(last).relu()
            last = self.dropout_layer(last)

        out = self.out_layer(last).squeeze()
        return out