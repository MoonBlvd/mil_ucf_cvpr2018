import torch
from trainer import do_train
from torch.utils import data
from pred_head import PredHead
from A3D_MIL_dataset import A3DMILDataset
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from sklearn import metrics
import numpy as np
import pdb
from torch.nn.utils.rnn import pad_sequence

torch.set_printoptions(precision=2)
wandb.init()

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:1" if use_cuda else "cpu")
# cudnn.benchmark = True


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens


# Parameters
params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1, 'collate_fn': pad_collate}

# params = {'batch_size': 4, 'shuffle': True, 'num_workers': 1}
val_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1, 'collate_fn': pad_collate}
# val_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1}
max_epochs = 2000

training_set = A3DMILDataset('/home/data/vision7/A3D_feat/dataset/train/',
                             batch_size=1,
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


def loss_fn(outputs, labels, len_outputs, len_labels):
    batch_size = outputs.size()[0]
    mask = torch.zeros(outputs.view(batch_size, -1).shape).to(device)
    for i, l in enumerate(len_outputs):
        mask[i, :l] = 1.0
    normal_max = torch.max(outputs.view(batch_size, -1) * mask *
                           (1.0 - labels.view(batch_size, -1)),
                           dim=1).values
    abnormal_max = torch.max(outputs.view(batch_size, -1) * mask * labels.view(batch_size, -1),
                             dim=1).values
    loss = torch.mean(1.0 - abnormal_max + normal_max)
    return torch.max(torch.tensor(0.0).to(device), loss.float())
    # return torch.max(torch.tensor(0.0).to(device), 1.0 - abnormal_max + normal_max)


optimizer = torch.optim.Adagrad(net.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001 )
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
for epoch in range(max_epochs):
    # Training
    net.train()
    running_loss = 0.0
    for idx, (
            local_batch,
            local_labels,
            len_batch,
            len_labels,
    ) in enumerate(tqdm(data_loader)):
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        # print(local_batch.shape)
        optimizer.zero_grad()
        outputs = net(local_batch)
        loss = loss_fn(outputs.view(outputs.size()[0], -1), local_labels, len_batch, len_labels)
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
        if idx % 10 == 0:
            wandb.log({"training loss": running_loss})
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
            for idx, (batch, label, len_batch, len_label) in enumerate(val_dataloader):
                batch = batch.to(device)
                label = label.to(device)
                outputs = net(batch)
                loss = loss_fn(outputs, label, len_batch, len_label)
                ones = torch.ones(outputs.shape).to(device)
                zeros = torch.zeros(outputs.shape).to(device)
                predicted = outputs.squeeze(-1)
                all_one_label = ones.squeeze(-1)
                test_running_loss += (loss.item() - test_running_loss) / (idx + 1)
                # for p, y in zip(predicted.cpu(), label.cpu()):
                for p, y, one in zip(predicted, label, all_one_label):
                    y_ct.append(y)
                    pred_ct.append(p)
                    all_one_ct.append(one)
                if idx == 0:
                    print('==============val sample==============')
                    # print(names)
                    print(p)
                    print(y)
            # print(y_ct)
            y_ct = torch.cat(y_ct, dim=0)
            pred_ct = torch.cat(pred_ct, dim=0)
            all_one_ct = torch.cat(all_one_ct, dim=0)
            y_ct = y_ct.cpu().numpy()
            pred_ct = pred_ct.cpu().numpy()
            all_one_ct = all_one_ct.cpu().numpy()
            one_fpr, one_tpr, one_thresholds = metrics.roc_curve(y_ct, all_one_ct, pos_label=1)
            fpr, tpr, thresholds = metrics.roc_curve(y_ct, pred_ct, pos_label=1)
            print("all one auc: {}".format(metrics.auc(one_fpr, one_tpr)))
            print("auc: {}".format(metrics.auc(fpr, tpr)))
            # print(y_ct.shape, pred_ct.shape)
            wandb.log({"validation loss": test_running_loss})
            wandb.log({"auc": metrics.auc(fpr, tpr)})
