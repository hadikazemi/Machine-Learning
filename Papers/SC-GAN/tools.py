from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import itertools


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


##### My Codes #####


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class VGGLoss(nn.Module):
    def __init__(self, start_lay):
        super(VGGLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.start_lay = start_lay

    def forward(self, x_vgg, x_rec_vgg):
        loss = 0
        for i in range(self.start_lay, len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_rec_vgg[i], x_vgg[i].detach())
        return loss


class StyleLoss(nn.Module):
    def __init__(self, lays):
        super(StyleLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.lays = lays
        self.gram = GramMatrix()

    def forward(self, x_vgg1, x_vgg2):
        loss = 0
        for lay in self.lays:
            gram1 = self.gram.forward(x_vgg1[lay])
            gram2 = self.gram.forward(x_vgg2[lay])
            loss += 0.5 * torch.mean(torch.abs(gram1 - gram2))
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class PerceptualDiscriminator(nn.Module):
    def __init__(self, input_nc=128, ndf=64, norm_layer=nn.BatchNorm2d, start_lay=1):
        super(PerceptualDiscriminator, self).__init__()
        use_bias = False

        kw = 4
        padw = 1
        self.start_lay = start_lay

        if start_lay == 1:
            self.seq1 = nn.Sequential(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            )  # 16x16

        if start_lay < 3:
            seq2_size = 64 + 256
            if start_lay==2:
                seq2_size = 256
            self.seq2 = nn.Sequential(
                nn.Conv2d(seq2_size, 64, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(64),
                nn.LeakyReLU(0.2, True)
            )  # 8x8

        seq3_size = 64 + 512
        if start_lay==3:
            seq3_size = 512

        self.seq3 = nn.Sequential(
            nn.Conv2d(seq3_size, 64, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True)
        )  # 8x8
        self.seq4 = nn.Sequential(
            nn.Conv2d(64 + 512, 64, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 4x4
            nn.Conv2d(64, 1, kernel_size=2, stride=1, padding=0)  # 1x1
        )
        # self.model = nn.Sequential(*sequence)

    def forward(self, layers):
        h_relu2, h_relu3, h_relu4, h_relu5 = layers[1], layers[2], layers[3], layers[4]
        if self.start_lay == 1:
            x = self.seq1(h_relu2)  # 16x16
            x = torch.cat([h_relu3, x], 1)
            x = self.seq2(x)  # 8x8
            x = torch.cat([h_relu4, x], 1)
            x = self.seq3(x)  # 4x4
        elif self.start_lay == 2:
            x = h_relu3
            x = self.seq2(x)
            x = torch.cat([h_relu4, x], 1)
            x = self.seq3(x)
        elif self.start_lay == 3:
            x = h_relu4
            x = self.seq3(x)

        x = torch.cat([h_relu5, x], 1)
        return self.seq4(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, img_size=64):
        super(discriminator, self).__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        if img_size == 128:
            self.conv32 = nn.Conv2d(d * 4, d * 4, 4, 2, 1)
            self.conv32_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        if self.img_size == 128:
            x = F.leaky_relu(self.conv32_bn(self.conv32(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x)

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class generator_old(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator_old, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x


class generator(nn.Module):
    def __init__(self, n_res=4, d=128, res_norm='adain', activ='relu', pad_type='zero', s_dim=0, mlp_dim=256, c_size=60, img_size=64):
        super(generator, self).__init__()

        self.model = []
        activation = nn.ReLU()
        self.model += [nn.ConvTranspose2d(c_size, d*8, 4, 1, 0), activation, nn.BatchNorm2d(d * 8)]
        self.model += [nn.ConvTranspose2d(d*8, d*4, 4, 2, 1), activation, nn.BatchNorm2d(d * 4)]
        self.model += [nn.ConvTranspose2d(d*4, d*2, 4, 2, 1), activation, nn.BatchNorm2d(d * 2)]
        self.model += [nn.ConvTranspose2d(d*2, d, 4, 2, 1), activation, nn.BatchNorm2d(d)]
        if img_size == 128:
            self.model += [nn.ConvTranspose2d(d, d, 4, 2, 1), activation, nn.BatchNorm2d(d)]
        self.model += [ResBlocks(n_res, d, res_norm, activ, pad_type=pad_type)]  # AdaIN residual blocks
        self.model += [nn.ConvTranspose2d(d, 3, 4, 2, 1), nn.Tanh()]
        self.model = nn.Sequential(*self.model)

        self.mlp = MLP(s_dim, self.get_num_adain_params(self.model), mlp_dim, 3)

    def forward(self, z, s):
        adain_params = self.mlp(s)
        self.assign_adain_params(adain_params)
        images = self.model(z)
        return images

    def assign_adain_params(self, adain_params):
        # assign the adain_params to the AdaIN layers in model
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def show_result(G, num_epoch, show = False, save=False, path = 'result.png', isFix=False, fixed_z_=None, gpu_id=0):
    z_ = torch.zeros(0, 100, 1, 1)
    # for i in range(5):
    #     c_ = torch.randn((1, 60)).view(-1, 60, 1, 1)
    #     for j in range(5):
    #         s_ = torch.randn((1, 40)).view(-1, 40, 1, 1)
    #         z_ = torch.cat((z_, torch.cat((c_, s_), 1)), 0)
    s_ = torch.randn((5, 40)).view(-1, 40, 1, 1)
    for i in range(5):
        c_ = torch.randn((1, 60)).view(-1, 60, 1, 1)
        for j in range(5):
            z_ = torch.cat((z_, torch.cat((c_, s_[j:j+1]), 1)), 0)
    z_ = Variable(z_.cuda(gpu_id), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_[:, :60], fixed_z_[:, 60:])
    else:
        test_images = G(z_[:, :60], z_[:, 60:])
    G.train()

    my_dpi = 96
    # plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def load_state(model, file):
    pretrained_dict = torch.load(file)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
