import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.utils.tensorboard import SummaryWriter
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Neural Network running on {device}")

class Convolutional(nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        #self.conv3 = nn.Conv2d(64, 128, 3)

        x = torch.randn(28, 28).view(-1, 1, 28, 28)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 10)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            print("Flatten to linear:", self._to_linear)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)



























if __name__ == '__main__':

    #writer = SummaryWriter("runs/convolutional")
    
    net = Convolutional()
    with torch.no_grad():
        x = torch.randn(28, 28).view(-1, 1, 28, 28)
        y = torch.randn(28, 28).view(-1, 1, 28, 28)
        
        import numpy as np
        a = net(x)
        b = torch.tensor(np.eye(10)[0], dtype=torch.float32)

        print(a)
        print(b)

        #loss = nn.MSELoss(a, a)
        loss = F.mse_loss(a,a)
        print(loss)

    writer.add_graph(net, x)
    writer.close()

