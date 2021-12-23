import torch
import torch.nn as nn
from pathlib import Path

def conv_2(c1,c2):
    lists=[
        nn.Conv2d(c1,c2,kernel_size=3,padding=1),
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU(),
        nn.Conv2d(c2,c2,kernel_size=3,padding=1),
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU()
    ]
    return nn.Sequential(*lists)
def conv_3(c1,c2):
    lists=[
        nn.Conv2d(c1,c2,3,padding=1),
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU(),
        nn.Conv2d(c2,c2,3,padding=1),
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU(),
        nn.Conv2d(c2,c2,3,padding=1),
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU()
    ]
    return nn.Sequential(*lists)
def deconv_3(c1,c2):
    return nn.Sequential(
        nn.ConvTranspose2d(c1,c2,3,padding=1), 
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU(),
        nn.ConvTranspose2d(c2,c2,3,padding=1), 
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU(),
        nn.ConvTranspose2d(c2,c2,3,padding=1), 
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU()
    )
def deconv_2(c1,c2):
    return nn.Sequential(
        nn.ConvTranspose2d(c1,c2,3,padding=1), 
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU(),
        nn.ConvTranspose2d(c2,c2,3,padding=1), 
        nn.BatchNorm2d(c2, track_running_stats=False),
        nn.ReLU()
    )
def addon(c1,c2):
    return nn.Sequential(
        nn.ConvTranspose2d(c1,c2,3,padding=1), 
        nn.ReLU(),
        nn.ConvTranspose2d(c2,c2,3,padding=1)
    )

class NormalEstimator(nn.Module):
    def __init__(self, pretrained_weight = None):
        super(NormalEstimator,self).__init__()
        self.conv1 = conv_2(3,64)
        self.conv2 = conv_2(64,128)
        self.conv3 = conv_3(128,256)
        self.conv4 = conv_3(256,512)
        self.conv5 = conv_3(512,512)
        self.pool = nn.MaxPool2d(2,2,return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv5 = deconv_3(512,512)
        self.deconv4 = deconv_3(1024,256)
        self.deconv3 = deconv_3(512,128)
        self.deconv2 = deconv_2(256,64)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,3,padding=1)
        )
        cur_file_dir = Path(__file__).parent
        if pretrained_weight is not None:
            weights = torch.load(pretrained_weight, map_location='cpu')
            self.load_state_dict(weights)

    def forward(self,x):
        x1 = self.conv1(x)
        x2,idx1 = self.pool(x1)
        x2 = self.conv2(x2)
        x3,idx2 = self.pool(x2)
        x3 = self.conv3(x3)
        x4,idx3 = self.pool(x3)
        x4 = self.conv4(x4)
        x5,idx4 = self.pool(x4)
        x5 = self.conv5(x5)
        dx5 = self.deconv5(x5)
        dx4 = self.unpool(dx5,idx4, output_size=x4.size())
        dx4 = self.deconv4(torch.cat([dx4,x4],dim=1))
        dx3 = self.unpool(dx4,idx3, output_size=x3.size())
        dx3 = self.deconv3(torch.cat([dx3,x3],dim=1))
        dx2 = self.unpool(dx3,idx2, output_size=x2.size())
        dx2 = self.deconv2(torch.cat([dx2,x2],dim=1))
        dx1 = self.unpool(dx2,idx1, output_size=x1.size())
        x_out = self.deconv1(torch.cat([dx1,x1],dim=1))
        n = nn.functional.normalize(x_out,dim=1)
        normal = n[:, [0, 2, 1]]
        normal[:, 1] *= -1
        return normal
        