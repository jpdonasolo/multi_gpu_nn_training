import torch.nn as nn


class CNNClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.convolutions = nn.Sequential(
                nn.Conv2d( 
                    in_channels=1, 
                    out_channels=16,
                    kernel_size=5, 
                    padding="same"
                ),
                nn.Dropout(p=.2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d( 
                    in_channels=16, 
                    out_channels=32,
                    kernel_size=3, 
                    padding="same"
                ),
                nn.Dropout(p=.2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )

        self.classification = nn.Sequential(
                nn.Flatten(),
                nn.Linear(7**2 * 32, 128),
                nn.Dropout(p=.2),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(dim=-1)
            )            

    def forward(self, x):
        x = self.convolutions(x)
        y = self.classification(x)
        return y