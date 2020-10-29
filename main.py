from utils import Dataset, rechorus_loss, evaluate_method, collate, init_weights
from model import NNCF
import torch
from torch.utils.data import DataLoader
#import nni


torch.manual_seed(2020)
#tuner_params = nni.get_next_parameter()
''''''
tuner_params={
    "batch_size":128,
    "epoch":100,
    "patience":15,

    "embed_size": 16,
    "hidden_size": 16,
    "dropout": 0.5,
    "lr": 0.00044711047610543483,
    "l2": 0.000021596339273145315,
    "resolution": 0.7504085532980395,
    "neg_num": 2,

    "conv_kernel_size": 5,
    "pool_kernel_size": 5,
    "conv_out_channels": 1,
}
print('tuner_params',tuner_params)
epoch = tuner_params['epoch']
batch_size = tuner_params['batch_size']
patience =tuner_params['patience']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ml-100k, Grocery_and_Gourmet_Food, ml-1m
dataset = Dataset('ml-100k', tuner_params)
model = NNCF(dataset, tuner_params).to(device)
print(model)
model.apply(init_weights)
loss_fn = rechorus_loss
optimizer = torch.optim.Adam(model.parameters(), lr=tuner_params['lr'], weight_decay=tuner_params['l2'])
metrics = ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10']
best_HR5, best_scores,best_diversity, pre_HR5 = 0, {},0, 0


ld = {key:DataLoader(dataset.feed_dict[key], collate_fn=collate,
                     batch_size=batch_size, shuffle=True, drop_last=True)
      for key in ['train', 'dev', 'test']}


for i in range(epoch):
    # train
    model.train()
    loss_avg = 0
    for feed_dict in ld['train']:
        optimizer.zero_grad()
        prediction = model(feed_dict, dataset)
        loss = loss_fn(prediction)
        loss.backward()
        loss_avg += loss_fn(prediction)
        optimizer.step()
    loss_avg /= len(ld['train'])
    print('Epoch: {}, loss:{} train'.format(i,loss_avg))
    # validation and test (test不用来选择参数)
    model.eval()
    phases = ['dev', 'test']
    vt_loss = {key:0 for key in phases}
    vt_res, vt_prediction, vt_res_tmp = {}, {}, {}

    for phase in phases:
        vt_res[phase] = {key: 0 for key in metrics}
        for feed_dict in ld[phase]:
            vt_prediction[phase] = model(feed_dict, dataset)
            vt_res_tmp[phase] = evaluate_method(vt_prediction[phase])
            vt_res[phase] = {key: vt_res[phase][key] + vt_res_tmp[phase][key] for key in metrics}
            vt_loss[phase] += loss_fn(vt_prediction[phase])

        vt_loss[phase] /= len(ld[phase])
        vt_res[phase] = {key: vt_res[phase][key] / len(ld[phase]) for key in metrics}
        print('Epoch: {}, loss:{}, {} {}'.format(i,vt_loss[phase],vt_res[phase],phase))
        #if phase == 'dev': nni.report_intermediate_result(vt_res[phase]['HR@5'])
    # save best
    if vt_res['dev']['HR@5'] > best_HR5:
        best_HR5 = vt_res['dev']['HR@5']
        best_scores = vt_res
    # patience
    if vt_res['dev']['HR@5'] < pre_HR5: patience -= 1
    if patience == 0: break
    pre_HR5 = vt_res['dev']['HR@5']
    dataset.neg_sample()

print(best_scores['dev'], "dev")
print(best_scores['test'], "test")
#nni.report_final_result(best_scores['test']['HR@5'])



