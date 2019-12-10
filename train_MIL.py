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

torch.set_printoptions(precision=2)
wandb.init()

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True

# Parameters
params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}

max_epochs = 10000

training_set = A3DMILDataset('/home/data/vision7/A3D_feat/dataset/train/',
                             batch_size=2,
                             phase='train')
data_loader = data.DataLoader(training_set, **params)
# test_loader = data.DataLoader(training_set, **params)

val_set = A3DMILDataset('/home/data/vision7/A3D_feat/dataset/train', batch_size=1, phase='val')
val_params = params.copy()
val_params["shuffle"] = False
val_dataloader = data.DataLoader(val_set, **val_params)

net = PredHead()
net.cuda()
wandb.watch(net)

for params in net.parameters():
    params.requires_grad = True


def loss_fn(outputs, labels):
    batch_size = outputs.size()[0]
    normal_max = torch.max(outputs.view(batch_size, -1) * (1.0 - labels.view(batch_size, -1)))
    abnormal_max = torch.max(outputs.view(batch_size, -1) * labels.view(batch_size, -1))
    return torch.max(torch.tensor(0.0).cuda(), 1.0 - abnormal_max + normal_max)


optimizer = torch.optim.Adagrad(net.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001 )
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
for epoch in range(max_epochs):
    # Training
    net.train()
    running_loss = 0.0
    for idx, (local_batch, local_labels) in enumerate(tqdm(data_loader)):
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        optimizer.zero_grad()
        outputs = net(local_batch)
        if idx % 1000 == 0:
            print(outputs, local_labels)
        loss = loss_fn(outputs, local_labels)
        # if idx % 10 == 0:
        # print(outputs, local_labels, loss)
        running_loss += (loss.item() - running_loss) / (idx + 1)
        loss.backward()
        optimizer.step()
        # if idx % 10 == 0:
        wandb.log({"training loss": running_loss})
        # if idx % 1000 == 0:
        # print(local_batch, local_labels)
    print("Epoch:{}. running loss: {:5f}".format(epoch, running_loss))

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
            for idx, (batch, label) in enumerate(val_dataloader):
                batch = batch.to(device)
                label = label.to(device)
                outputs = net(batch)
                loss = loss_fn(outputs, label)
                # if idx % 10 == 0:
                # print(outputs, label, loss)
                ones = torch.ones(outputs.shape).cuda()
                zeros = torch.zeros(outputs.shape).cuda()
                predicted = outputs.squeeze(-1)
                test_running_loss += (loss.item() - test_running_loss) / (idx + 1)
                # if idx % 10 == 0:
                # y = label.cpu().numpy()
                for p, y in zip(predicted.cpu(), label.cpu()):
                    # y_ct = torch.cat((y_ct, y), dim=0)
                    y_ct.append(y)
                    pred_ct.append(p)
                    # pred_ct = torch.cat((pred_ct, p), dim=0)
                if idx % 1000 == 0:
                    print(p)
                    print(y)
            y_ct = torch.cat(y_ct, dim=0)
            pred_ct = torch.cat(pred_ct, dim=0)
            y_ct = y_ct.numpy()
            pred_ct = pred_ct.numpy()
            fpr, tpr, thresholds = metrics.roc_curve(y_ct, pred_ct, pos_label=1)
            print("auc: {}".format(metrics.auc(fpr, tpr)))
            # print(y_ct.shape, pred_ct.shape)
            wandb.log({"validation loss": test_running_loss})
            wandb.log({"auc": metrics.auc(fpr, tpr)})
    # print(test_running_loss / (idx + 1))
    # print('Accuracy of the network on the epoch : %d %%' % (100 * correct / total))
