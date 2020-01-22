import torch
from trainer import do_train
from torch.utils import data
from pred_head import PredHead
from A3D_MIL_dataset_padding import A3DMILDataset
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from sklearn import metrics
import numpy as np
import pdb
from torch.nn.utils.rnn import pad_sequence
from loss import MILLoss

torch.set_printoptions(precision=2)
wandb.init()

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens
    # return xx_pad, yy_pad


def pad_collate_val(batch):
    (xx, yy, pad_num) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens
    # return xx_pad, yy_pad


# Parameters
params = {'batch_size': 4, 'shuffle': True, 'num_workers': 1, 'collate_fn': pad_collate}

# params = {'batch_size': 4, 'shuffle': True, 'num_workers': 1}
# val_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1, 'collate_fn': pad_collate_val}
val_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1}
max_epochs = 200

training_set = A3DMILDataset('/home/data/vision7/A3D_feat/dataset/train/',
                             batch_size=4,
                             phase='train')
data_loader = data.DataLoader(training_set, **params)
# test_loader = data.DataLoader(training_set, **params)

val_set = A3DMILDataset('/home/data/vision7/A3D_feat/dataset/val', batch_size=1, phase='val')
val_dataloader = data.DataLoader(val_set, **val_params)
# print(len(val_dataloader))

net = PredHead()
net.to(device)
wandb.watch(net)

for params in net.parameters():
    params.requires_grad = True


def loss_fn(outputs, labels, len_outputs, len_labels, phase='train', abnormal_pad_num=0):
    # def loss_fn(outputs, labels, phase='train', abnormal_pad_num=0):
    batch_size = outputs.size()[0]
    # if phase == 'val':
    # outputs = tile(outputs, dim=1, n_tile=abnormal_pad_num)
    if phase == 'train':
        mask = torch.zeros(outputs.view(batch_size, -1).shape).to(device)
        for i, l in enumerate(len_outputs):
            mask[i, :l] = 1.0
        normal_max = torch.max(outputs.view(batch_size, -1) * mask *
                               (1.0 - labels.view(batch_size, -1)),
                               dim=1).values.float()
        abnormal_max = torch.max(outputs.view(batch_size, -1) * mask * labels.view(batch_size, -1),
                                 dim=1).values.float()
    else:
        normal_max = torch.max(outputs.view(batch_size, -1) * (1.0 - labels.view(batch_size, -1)),
                               dim=1).values.float()
        abnormal_max = torch.max(outputs.view(batch_size, -1) * labels.view(batch_size, -1),
                                 dim=1).values.float()
    loss = torch.max(torch.tensor(0.0).to(device), 1.0 - abnormal_max + normal_max)
    # print(torch.flatten(outputs), labels, loss)
    # loss = torch.mean(1.0 - abnormal_max + normal_max)
    return torch.mean(loss), torch.flatten(outputs)
    # return torch.max(torch.tensor(0.0).to(device), 1.0 - abnormal_max + normal_max)


def mil_loss_fn(outputs, labels, phase='train', abnormal_pad_num=0, normal_pad_num=0):
    mil_loss = MILLoss()
    # normal_outputs = tile(outputs, dim=1, n_tile=normal_pad_num)
    # abnormal_outputs = tile(outputs, dim=1, n_tile=)
    if phase == 'val':
        abnormal_outputs = tile(outputs, dim=1, n_tile=abnormal_pad_num)
    loss = mil_loss.forward(labels, outputs)


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)


# optimizer = torch.optim.Adagrad(net.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001 )
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
iters = 0
for epoch in range(max_epochs):
    # Training
    net.train()
    running_loss = 0.0
    for idx, (local_batch, local_labels, len_outputs, len_labels) in enumerate(tqdm(data_loader)):
        # Transfer to GPU
        iters += 1
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        # print(local_batch.shape)
        optimizer.zero_grad()
        outputs = net(local_batch)
        loss, outputs = loss_fn(outputs.view(outputs.size()[0], -1), local_labels, len_outputs,
                                len_labels)
        if idx % 1000 == 0:
            print('===========training sample===============')
            # print(names)
            print(local_batch, local_labels)
            # print(loss)
        # if idx % 10 == 0:
        # print(outputs, local_labels, loss)
        running_loss += (loss.item() - running_loss) / (idx + 1)
        loss.backward()
        optimizer.step()
        if idx % 20 == 0:
            wandb.log({"training loss": running_loss}, step=iters)
        # if idx % 1000 == 0:
        # print(local_batch, local_labels)
    print("Epoch:{}. running loss: {:5f}".format(epoch, running_loss))
    training_set.callback()

    eval = True
    if eval:
        net.eval()
        # print(running_loss)
        correct = 0
        total = 0
        print("=========begin to eval==========")
        test_running_loss = 0.0
        with torch.no_grad():
            y_ct = []
            pred_ct = []
            all_one_ct = []
            for idx, (
                    batch,
                    label,
                    abnormal_pad_num,
            ) in enumerate(val_dataloader):
                batch = batch.to(device)
                label = label.to(device)
                outputs = net(batch)
                loss, outputs = loss_fn(outputs,
                                        label,
                                        len_outputs=0,
                                        len_labels=0,
                                        phase='val',
                                        abnormal_pad_num=abnormal_pad_num)
                ones = torch.ones(outputs.shape).to(device)
                predicted = outputs.squeeze(-1)
                all_one_label = ones.squeeze(-1)
                label = torch.flatten(label).squeeze(-1)
                test_running_loss += (loss.item() - test_running_loss) / (idx + 1)
                count = 0
                for p, y, one in zip(predicted, label, all_one_label):
                    count += 1
                    y_ct.append(y.item())
                    pred_ct.append(p.item())
                    all_one_ct.append(one.item())
                if idx == 0:
                    print('==============val sample==============')
                    # print(names)
                    print(count)
                    print(pred_ct)
                    print(y_ct)
                    print(all_one_ct)
            # print(y_ct)
            # y_ct = torch.cat(y_ct, dim=0)
            # pred_ct = torch.cat(pred_ct, dim=0)
            # all_one_ct = torch.cat(all_one_ct, dim=0)
            # y_ct = y_ct.numpy()
            # pred_ct = pred_ct.numpy()
            # all_one_ct = all_one_ct.numpy()
            y_ct = np.asarray(y_ct)
            pred_ct = np.asarray(pred_ct)
            all_one_ct = np.asarray(all_one_ct)
            print(pred_ct)
            one_fpr, one_tpr, one_thresholds = metrics.roc_curve(all_one_ct, pred_ct, pos_label=1)

            fpr, tpr, thresholds = metrics.roc_curve(y_ct, pred_ct, pos_label=1)
            print("all one auc: {}".format(metrics.auc(one_fpr, one_tpr)))
            print("auc: {}".format(metrics.auc(fpr, tpr)))
            # print(y_ct.shape, pred_ct.shape)
            wandb.log({"validation loss": test_running_loss}, step=iters)
            wandb.log({"auc": metrics.auc(fpr, tpr)}, step=iters)
