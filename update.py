import torch
import torch.nn as nn
import torch.nn.functional as F


class PredHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(PredHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, True) # nn.ReLU(inplace=True)
        self.outact = nn.LeakyReLU(0.2, True)  # nn.Tanh()

    def forward(self, x):
        return self.outact(self.conv2(self.relu(self.conv1(x))))
        # return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()
        self.convf1 = nn.Conv2d(3, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64-3, 3, padding=1)

    def forward(self, flow):
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        return torch.cat((flo, flow), dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=32):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=64, input_dim=64+64)
        self.pre_head = PredHead(64, hidden_dim=128)

    def forward(self, net, inp, rec_image):
        rec_image_features = self.encoder(rec_image)
        # print(inp.shape, rec_image_features.shape)
        inp = torch.cat([inp, rec_image_features], dim=1)
        # print(net.shape, inp.shape)
        net = self.gru(net, inp)   # net是hidden state，inp是fk-1
        drec_image = self.pre_head(net)

        return net, drec_image, rec_image_features, inp
