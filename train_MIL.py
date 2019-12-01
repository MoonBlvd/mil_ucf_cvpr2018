import torch
from trainer import do_train
from torch.utils import data
from pred_head import PredHead
from A3D_MIL_dataset import A3DMILDataset
from tqdm import tqdm
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}

max_epochs = 10000

training_set = A3DMILDataset('/home/data/vision7/A3D_frame_feat/', batch_size=2)
data_loader = data.DataLoader(training_set, **params)
test_loader = data.DataLoader(training_set, **params) 

net = PredHead() 
net.cuda() 
for params in net.parameters():
    params.requires_grad = True

def loss_fn(outputs, labels):
    batch_size = outputs.size()[0]
    normal_max = torch.max(outputs.view(1, 1, -1) *(1.0-labels))
    abnormal_max = torch.max(outputs.view(1,1,-1) *labels)
    return torch.max(torch.tensor(0.0).cuda(), 1.0-abnormal_max+normal_max)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)
# optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001 )

for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    net.train() 
    for idx, (local_batch, local_labels) in enumerate(tqdm(data_loader)):
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        optimizer.zero_grad()
        outputs = net(local_batch)
        loss = loss_fn(outputs, local_labels)
        # if idx % 10 == 0:
            # print(outputs, local_labels, loss) 
        running_loss += (loss.item() - running_loss) / (idx+1)
        loss.backward()
        optimizer.step()
    print(running_loss) 
    correct = 0
    total = 0
    print("=========begin to eval==========") 
    test_running_loss = 0
    with torch.no_grad():
        net.eval() 
        for idx,(batch,label) in enumerate(data_loader):
            batch = batch.to(device)
            label = label.to(device)  
            outputs = net(batch)
            loss = loss_fn(outputs, label)
            if idx % 10 == 0:
                print(outputs, label,loss) 
            outputs = outputs.view(1,1,-1) 
            ones = torch.ones(outputs.shape).cuda() 
            zeros = torch.zeros(outputs.shape).cuda() 
            predicted = torch.where(outputs>0.5, ones, zeros) 
            total += label.size(1)
            correct += (predicted == label).sum().item()
            test_running_loss += (loss.item() - test_running_loss) / (idx+1)
    print(test_running_loss) 
    print('Accuracy of the network on the epoch : %d %%' % (100* correct / total) ) 
