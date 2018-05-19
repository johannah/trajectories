# learning pixelcnn -
# largely based on code from Ritesh Kumar - https://github.com/ritheshkumar95/pytorch-vqvae/
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from IPython import embed
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init_xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            pass
            #print('not initializing {}'.format(classname))

class GatedActivation(nn.Module):
    def __init__(self):
        super(GatedActivation, self).__init__()

    def forward(self, x):
        x,y = x.chunk(2,dim=1)
        return F.tanh(x)*F.sigmoid(y)

class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10, cond_size=None):
        super(GatedMaskedConv2d, self).__init__()
        assert (kernel % 2 == 1 )
        print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        # unique for every layer of the pixelcnn - takes integer from 0-x and
        # returns slice which is 0 to 1ish - treat each latent value like a
        # "word" embedding
        self.class_cond_embedding = nn.Embedding(n_classes, 2*dim)
        vkernel_shape = (kernel//2 + 1, kernel)
        vpadding_shape = (kernel//2, kernel//2)

        cond_kernel_shape = (kernel, kernel)
        cond_padding_shape = (kernel//2, kernel//2)

        hkernel_shape = (1,kernel//2+1)
        hpadding_shape = (0,kernel//2)

        if cond_size is not None:
            self.spatial_cond_stack = nn.Conv2d(cond_size, dim*2, kernel_size=cond_kernel_shape, stride=1, padding=cond_padding_shape)

        self.vert_stack = nn.Conv2d(dim, dim*2, kernel_size=vkernel_shape, stride=1, padding=vpadding_shape)
        self.vert_to_horiz = nn.Conv2d(2*dim, 2*dim, kernel_size=1)
        self.horiz_stack = nn.Conv2d(dim, dim*2, kernel_size=hkernel_shape, stride=1, padding=hpadding_shape)
        # kernel_size 1 are "fixup layers" to make things match up
        self.horiz_resid = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:,:,-1].zero_() # mask final row
        self.horiz_stack.weight.data[:,:,:,-1].zero_() # mask final column

    def forward(self, x_v, x_h, class_condition=None, spatial_condition=None):
        # class condition coming in is just an integer
        # spatial_condition should be the same size as the input
        if self.mask_type == 'A':
            # make first layer causal to prevent cheating
            self.make_causal()
        # manipulation to get same size out of h_vert
        # output of h_vert is 6,6,(2*dim)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:,:,:x_v.size(-1), :]
        # h_vert is (batch_size,512,6,6)

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:,:,:,:x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        input_to_out_v = h_vert
        input_to_out_h = v2h + h_horiz

        # add class conditioning
        if class_condition is not None:
            class_condition = self.class_cond_embedding(class_condition)
            input_to_out_v += class_condition[:,:,None,None]
            input_to_out_h += class_condition[:,:,None,None]

        if spatial_condition is not None:
            spatial_c_e = self.spatial_cond_stack(spatial_condition)
            input_to_out_v += spatial_c_e
            input_to_out_h += spatial_c_e

        out_v = self.gate(input_to_out_v)
        gate_h = self.gate(input_to_out_h)

        if self.residual:
            out_h = self.horiz_resid(gate_h)+x_h
        else:
            out_h = self.horiz_resid(gate_h)
        return out_v, out_h

class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=512, dim=256, n_layers=15, n_classes=10, spatial_cond_size=None):
        super(GatedPixelCNN, self).__init__()
        if spatial_cond_size is None:
            scond_size = 'na'
        else:
            scond_size = spatial_cond_size
        # input_dim is the size of all possible values (in vqvae should be
        # num_clusters)
        self.name = 'rpcnn_id%d_d%d_l%d_nc%d_cs%s'%(input_dim, dim, n_layers, n_classes, scond_size)
        self.dim = dim
        # lookup table to store input
        self.embedding = nn.Embedding(input_dim, self.dim)

        if spatial_cond_size is not None:
            # assume same vocab size - but input_dim could be different here
            self.spatial_cond_embedding = nn.Embedding(input_dim, self.dim)
        # build pixelcnn layers - functions like normal python list, but modules are registered
        self.layers = nn.ModuleList()
        # first block has Mask-A convolution - (no residual connections)
        # subsequent blocks have Mask-B convolutions
        self.layers.append(GatedMaskedConv2d(mask_type='A', dim=self.dim,
                           kernel=7, residual=False, n_classes=n_classes, cond_size=spatial_cond_size))
        for i in range(1,n_layers):
            self.layers.append(GatedMaskedConv2d(mask_type='B', dim=self.dim,
                           kernel=3, residual=True, n_classes=n_classes, cond_size=spatial_cond_size))

        self.output_conv = nn.Sequential(
                                         nn.Conv2d(self.dim, 512, 1),
                                         nn.ReLU(True),
                                         nn.Conv2d(512, input_dim, 1)
                                         )
        # in pytorch - apply(fn)  recursively applies fn to every submodule as returned by .children
        # apply xavier_uniform init to all weights
        self.apply(weights_init)

    def forward(self, x, label=None, spatial_cond=None):
        # x is (B,H,W,C)
        shp = x.size()+(-1,)
        xo = self.embedding(x.contiguous().view(-1)).view(shp)
        # change order to (B,C,H,W)
        xo = xo.permute(0,3,1,2)
        x_v, x_h = (xo,xo)

        if spatial_cond is not None:
            # coming in, spatial_cond is (batch_size,  frames, 6, 6)
            sshp = spatial_cond.size()+(-1,)
            sxo = self.spatial_cond_embedding(spatial_cond.contiguous().view(-1)).view(sshp)
            # now it is  (batch_size, frames_hist, 6, 6, 256)
            spatial_cond = sxo.permute(0,2,3,1,4).contiguous()
            sc_shp = spatial_cond.shape
            spatial_cond = spatial_cond.reshape(sc_shp[:-2]+(-1,))
            spatial_cond = spatial_cond.permute(0,3,1,2)

        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label, spatial_cond)
        return self.output_conv(x_h)

    def generate(self, label=None, spatial_cond=None, shape=(8,8), batch_size=1):

        param = next(self.parameters())
        x = torch.zeros(
                (batch_size, shape[0], shape[1]),
                dtype=torch.int64, device=param.device)

        if spatial_cond is not None:
            # batch size and spatial cond batch size must be the same
            batch_size = spatial_cond.shape[0]

        if batch_size != 1:
            raise ValueError('generator needs 1 size now TODO - fix')

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label=label, spatial_cond=spatial_cond)
                #probs = F.softmax(logits[:,:,i,j], -1)
                #x.data[:,i,j].copy_(probs.multinomial(1).squeeze().data)
                x.data[:,i,j].copy_(torch.argmax(logits[:,:,i,j]))
        return x


