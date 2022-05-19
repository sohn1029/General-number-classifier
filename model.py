import torch.nn as nn


class RobustModel(nn.Module):

    def __init__(self):
        super(RobustModel, self).__init__()

        self.keep_prob = 0.5

        self.layer1 = nn.Sequential(
            #28 28 1
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            #26 26 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32))
            #13 13 32
        
        self.fc1 = nn.Linear(13 * 13 * 32, 110, bias=True)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')

        self.layer1_2 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(110),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))


        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            #12 12 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64))
        #6 6 64

        self.fc2 = nn.Linear(6 * 6 * 64, 110, bias=True)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        
        self.layer2_2 = nn.Sequential(
            self.fc2,
            nn.BatchNorm1d(110),
            nn.ReLU(),
            nn.Dropout(p=1- self.keep_prob))


        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #6 6 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2, padding = 1),
            nn.BatchNorm2d(128))
        #4 4 128

        self.fc3 = nn.Linear(4 * 4 * 128, 110, bias=True)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        
        self.layer3_2 = nn.Sequential(
            self.fc3,
            nn.BatchNorm1d(110),
            nn.ReLU(),
            nn.Dropout(p=1- self.keep_prob))
        #2 2 128
        
        self.fc4 = nn.Linear(110, 10)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
    

    def forward(self, x):

        a1 = self.layer1(x)
        a1_2 = a1.view(a1.size(0), -1)
        a1_2 = self.layer1_2(a1_2)

        a2 = self.layer2(a1)
        a2_2 = a2.view(a2.size(0), -1)
        a2_2 = self.layer2_2(a2_2)

        a3 = self.layer3(a2)
        a3_2 = a3.view(a3.size(0), -1)
        a3_2= self.layer3_2(a3_2)
        
        a4 = (a1_2 + a2_2 + a3_2)/3

        y = self.fc4(a4)

        return y

