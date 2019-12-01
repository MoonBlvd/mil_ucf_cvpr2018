import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)  

class PredHead(nn.Module):
    def __init__(self):
        super(PredHead, self).__init__()
        self.model = nn.Sequential(nn.Linear(1024, 512),
                                nn.ReLU(),
                                # nn.Dropout(0.9),
                                nn.Linear(512, 32),
                                nn.ReLU(),
                                # nn.Dropout(0.9),
                                nn.Linear(32, 1),
                                nn.Sigmoid(),
                                )

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
