# learning pixelcnn - 
# largely based on code from Ritesh Kumar - https://github.com/ritheshkumar95/pytorch-vqvae/
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

def weights_init(mod):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init_xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print('not initializing {}'.format(classname))

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x,y = x.chunk(2,dim=1)
        return F.tanh(x)*F.sigmoid(y)

class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert (kernel % 2 == 1 ) 
        print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        self.class_cond_embedding = nn.Embedding(n_classes, 2*dim)
        vkernel_shape = (kernel//2 + 1, kernel)
        vpadding_shape = (kernel//2, kernel//2)
        hkernel_shape = (1,kernel//2+1)
        hpadding_shape = (0,kernel//2)

        self.vert_stack = nn.Conv2d(dim, dim*2, vkernel_shape, vpadding_shape)
        self.vert_to_horz = nn.Conv2d(2*dim,2*dim,1)
        self.horiz_stack = nn.Conv2d(dim, dim*2, hkernel_shape, hpadding_shape)
        self.horiz_resid = nn.Conv2d(dim, dim, 1)
        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:,:,-1].zero_() # mask final row
        self.horiz_stack.weight.data[:,:,:,-1].zero_() # mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            # make first layer causal to prevent cheating
            self.make_causal()
        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_ver = h_vert[:,:,:x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:,:,None,None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = _horiz[:,:,:,:x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz+h[:,:,None,None])
        if self.residual:
            out_h = self.horiz_resid(out)+x_h
        else:
            out_h = self.horiz_resid(out)
        return out_v, out_h

class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim
        # lookup table to store input
        self.embedding = nn.Embedding(input_dim, self.dim)
        # build pixelcnn layers - functions like normal python list, but modules are registered
        self.layers = nn.ModuleList()
        # first block has Mask-A convolution - (no residual connections)
        # subsequent blocks have Mask-B convolutions
        self.layers.append(GatedMaskedConv2d(mask_type='A', dim=self.dim, 
                           kernel=7, residual=False, n_classes=n_classes))
        for i in range(1,n_layers):
            self.layers.append(GatedMaskedConv2d(mask_type='B', dim=self.dim, 
                           kernel=3, residual=True, n_classes=n_classes))

        self.output_conv = nn.Sequential(
                                         nn.Conv2d(self.dim, 512, 1), 
                                         nn.ReLU(True), 
                                         nn.Conv2d(512, input_dim, 1)
                                         )
        # in pytorch - apply(fn)  recursively applies fn to every submodule as returned by .children
        # apply xavier_uniform init to all weights
        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size()+(-1,)
        # (B,H,W,C)
        xo = self.embedding(x.view(-1)).view(shp) 
        # change order to (B,C,H,W)
        xo = xo.permute(0,3,1,2)
        x_v, x_h = (xo,xo)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)
        return self.output_conv(x_h)

    def generate(self, label, shape=(8,8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
                (batch_size, shape[0], shape[1]), 
                dtype=torch.int64, device=param.device)
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:,:,i,j], -1)
                x.data[:,i,j].copy_(
                        probs.multinomial(1).squeeze().data)
        return x


