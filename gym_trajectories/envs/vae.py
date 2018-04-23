# original vae code from https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(Encoder, self).__init__()
        # make encoder one layer bigger then decoder - kk
        self.dout = D_out
        self.linear1 = torch.nn.Linear(D_in, 1024)
        self.linear2 = torch.nn.Linear(1024, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.linear4 = torch.nn.Linear(256, self.dout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return F.relu(self.linear4(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(Decoder, self).__init__()
        self.latent_dim = D_in
        self.linear1 = torch.nn.Linear(self.latent_dim, 512)
        self.linear2 = torch.nn.Linear(512, 1024)
        self.linear3 = torch.nn.Linear(1024, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.relu(self.linear3(x))

class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder, use_cuda=False):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_cuda = use_cuda

        self._enc_mu = torch.nn.Linear(self.encoder.dout, self.decoder.latent_dim)
        self._enc_log_sigma = torch.nn.Linear(self.encoder.dout, self.decoder.latent_dim)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma
        if self.use_cuda:
            v = Variable(std_z, requires_grad=False).cuda()
        else:
            v = Variable(std_z, requires_grad=False)

        return mu + sigma * v  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':

    input_dim = 28 * 28
    batch_size = 32

    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print('Number of samples: ', len(mnist))

    latent_size = 64
    encoder = Encoder(input_dim, latent_size)
    decoder = Decoder(latent_size, input_dim)
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
            optimizer.zero_grad()
            dec = vae(inputs)
            from IPython import embed; embed()
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.data[0]
        print(epoch, l)

    plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)
