import torch
import torch.nn as nn
    
class PredHead(nn.Module):
    def __init__(self):
        super(EncoderGRU, self).__init__()
        model = nn.Sequential(nn.Linear(4096, 512),
                                nn.ReLU(),
                                nn.Dropout(0.6),
                                nn.Linear(512, 32),
                                nn.ReLU(),
                                nn.Dropout(0.6),
                                nn.Linear(32, 1),
                                nn.Sigmoid(),
                                )

        torch.nn.init.xavier_normal_(model.weight)

    def forward(self, x):
        return model(x)
