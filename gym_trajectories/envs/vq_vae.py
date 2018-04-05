# baseline learning vq-vae
# very strongly referenced vq-vae code from Ritesh Kumar from below link:
# https://github.com/ritheshkumar95/vq-vae-exps/blob/master/vq-vae/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from IPython import embed

def to_scalar(arr):
    if type(arr) == list:
        return [x.cpu().data.tolist()[0] for x in arr]
    else:
        return arr.cpu().data.tolist()[0]

class AutoEncoder(nn.Module):
    def __init__(self, num_clusters=512):
        super(AutoEncoder, self).__init__()
        data_channels_size = 1
        encoder_output_size = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=data_channels_size,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,
                      out_channels=encoder_output_size,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(encoder_output_size),
            )
        ## vq embedding scheme
        self.embedding = nn.Embedding(num_clusters, encoder_output_size)
        # common scaling for embeddings - variance roughly scales with num_clusters
        self.embedding.weight.data.copy_(1./num_clusters *
                                torch.randn(num_clusters,encoder_output_size))

        self.decoder = nn.Sequential(
                nn.Conv2d(in_channels=encoder_output_size,
                          out_channels=32,
                          kernel_size=1,
                          stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                # applies a 2d transposed convolution operator over input image
                # composed of several input planes. Can be seen as gradient of Conv2d
                # with respsct to its input. also known as fractionally-strided conv.
                nn.ConvTranspose2d(in_channels=32,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=16,
                        out_channels=data_channels_size,
                        kernel_size=4,
                        stride=2, padding=1),
                nn.Sigmoid()
                )

    def forward(self, x):
        # get continuous output directly from encoder
        z_e_x = self.encoder(x)
        # NCHW is the order in the encoder
        # (num, channels, height, width)
        N, C, H, W = z_e_x.size()
        # need NHWC instead of default NCHW for easier computations
        z_e_x_transposed = z_e_x.permute(0,2,3,1)
        # needs C,K
        emb = self.embedding.weight.transpose(0,1)
        # broadcast to determine distance from encoder output to clusters
        # NHWC -> NHWCK
        measure = z_e_x_transposed.unsqueeze(4) - emb[None, None, None]
        # square each element, then sum over channels
        dists = torch.pow(measure, 2).sum(-2)
        # pytorch gives real min and arg min - select argmin
        # this is the closest k for each sample - Equation 1
        latents = dists.min(-1)[1]

        # look up cluster centers
        z_q_x = self.embedding(latents.view(latents.size(0), -1))
        # back to NCHW (orig) - now cluster centers
        z_q_x = z_q_x.view(N, H, W, C).permute(0, 3, 1, 2)
        # put quantized data through decoder
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_e_x, z_q_x

if __name__ == '__main__':
    use_cuda = False
    ysize, xsize = 40,40
    if use_cuda:
        model = AutoEncoder().cuda()
        x = Variable(torch.randn(32,1,ysize,xsize).cuda(), requires_grad=False)
    else:
        model = AutoEncoder()
        x = Variable(torch.randn(32,1,ysize,xsize), requires_grad=False)

    model.zero_grad()
    x_tilde, z_e_x, z_q_x = model(x)
    z_q_x.retain_grad()

    # losses
    loss1 = F.binary_cross_entropy(x_tilde, x)
    loss1.backward(retain_graph=True)
    # make sure that encoder is not receiving gradients - only train decoder
    assert model.encoder[-2].bias.grad is None
    model.embedding.zero_grad()
    # straight-thru trick to skip discrete zs
    z_e_x.backward(z_q_x.grad, retain_graph=True)
    # make sure embedding has no gradient
    assert model.embedding.weight.grad.sum().data.cpu().numpy()[0] == 0
    bias = deepcopy(model.encoder[-2].bias.grad.data)

    # detach is like stop gradient
    loss2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss2.backward(retain_graph=True)
    emb = deepcopy(model.embedding.weight.grad.data)
    assert (bias == model.encoder[-2].bias.grad.data).all() is True

    # commitment loss
    Beta = 0.25
    loss3 = Beta*F.mse_loss(z_e_x, z_q_x.detach())
    loss3.backward()
    assert (emb == model.embedding.weight.grad.data).all() is True

    print(loss1, loss2, loss3)


