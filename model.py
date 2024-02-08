from update import BasicUpdateBlock
from extractor import BasicEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepEraser(nn.Module):
    def __init__(self):
        super(DeepEraser, self).__init__()
        self.hidden_dim = hdim = 64
        self.context_dim = 64

        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.update_block = BasicUpdateBlock(hidden_dim=64)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image1, mask, iters=8, test_mode=False):
        image1 = image1.contiguous()
        
        image2 = torch.cat([image1, mask], dim = 1)

        hdim = self.hidden_dim
        cdim = self.context_dim
        fmap1 = self.fnet(image2)

        net, inp = torch.split(fmap1, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        
        rec_image0 = image1
        rec_image = image1
        image_list = []
        recf_list = []
        inpf_list = []
        for itr in range(iters):
            net, d_rec_image, rec_image_features, inpf = self.update_block(net, inp, rec_image)
            rec_image = rec_image0 + d_rec_image
            image_list.append(rec_image)
            recf_list.append(rec_image_features)
            inpf_list.append(inpf)

        if test_mode:
            return rec_image

        return image_list  




