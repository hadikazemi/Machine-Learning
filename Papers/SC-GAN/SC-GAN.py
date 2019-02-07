import os, time, sys
import matplotlib.pyplot as plt
import pickle
import imageio
import torch
import torch.nn as nn
from tools import generator, discriminator, PerceptualDiscriminator, Vgg19, VGGLoss, StyleLoss, show_train_hist, show_result, load_state
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image

data_name = 'Shoes'
data_dir = 'data/edges2shoes/DCGAN/'

# data_name = 'Bags03'
# data_dir = 'data/edges2handbags/DCGAN/'
#
# data_name = 'CelebA_New'
# data_dir = 'data/celebA/resized_celebA'
#
# data_name = 'Apple'
# data_dir = 'data/apple2orange/DCGAN'
# #
# data_name = 'Facade'
# data_dir = 'data/facades/DCGAN'
#
data_name = 'LSUN_Living'
data_dir = 'living_room_train'

margin_c = 0.2
margin_s = 1.0
weight_c = 0.1
weight_s = 4    # CelebA 500
layer_c = 4
layer_s = [0, 1, 2]
gpu_id = 1
with_dp = True
img_size = 64
decay = 1
style_loss = True
report_period = 50
save_period = 1000

# training parameters
batch_size = 32
lr = 0.0002
train_epoch = 100
continue_train = False

spec = 'margin = {}\nweight_c = {}\nweight_s = {}\nlayer_c = {}\nlayer_s = {}\nwith_dp = {}\ndecay = {}\nstyle_loss = {}'.format(margin_c, weight_c, weight_s, layer_c, layer_s, with_dp, decay, style_loss)
f = open('spec.txt', 'w')
f.write(spec)
f.close()


fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(gpu_id), volatile=True)

# data_loader
transform = transforms.Compose([transforms.Resize([img_size, img_size], Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

if 'LSUN' in data_name:
    from torchvision.datasets import LSUN
    dset = LSUN('data/LSUN', classes=[data_dir], transform=transform)
else:
    dset = datasets.ImageFolder(data_dir, transform)

train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)
# temp = plt.imread(train_loader.dataset.imgs[0][0])

# network
vgg = Vgg19()
vgg.cuda(gpu_id)

G = generator(d=128, mlp_dim=256, s_dim=40, img_size=img_size)
D = discriminator(128, img_size=img_size)

G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

if continue_train:
    load_state(G, "{}_MYGAN_results/generator_param.pkl".format(data_name))
    load_state(D, "{}_MYGAN_results/discriminator_param.pkl".format(data_name))
    print "\n Models are loaded ! \n"

G.cuda(gpu_id)
D.cuda(gpu_id)

# Binary Cross Entropy loss
BCE_loss = nn.MSELoss()
criterionVGG = VGGLoss(layer_c)
criterionStyle = StyleLoss(layer_s)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('{}_MYGAN_results'.format(data_name)):
    os.mkdir('{}_MYGAN_results'.format(data_name))
if not os.path.isdir('{}_MYGAN_results/Random_results'.format(data_name)):
    os.mkdir('{}_MYGAN_results/Random_results'.format(data_name))
if not os.path.isdir('{}_MYGAN_results/Fixed_results'.format(data_name)):
    os.mkdir('{}_MYGAN_results/Fixed_results'.format(data_name))

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('Training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= decay
        D_optimizer.param_groups[0]['lr'] /= decay
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= decay
        D_optimizer.param_groups[0]['lr'] /= decay
        print("learning rate change!")

    num_iter = 0

    epoch_start_time = time.time()
    for x_, _ in train_loader:
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_.cuda(gpu_id)), Variable(y_real_.cuda(gpu_id)), Variable(y_fake_.cuda(gpu_id))
        D_result = D(x_).squeeze()
        D_real_loss = 0.5 * torch.mean((D_result - y_real_) ** 2)

        c_ = Variable(torch.randn((mini_batch, 60)).view(-1, 60, 1, 1).cuda(gpu_id))
        s_ = Variable(torch.randn((mini_batch, 40)).view(-1, 40, 1, 1).cuda(gpu_id))
        G_result = G(c_, s_)

        D_result = D(G_result).squeeze()
        D_fake_loss = 0.5 * torch.mean((D_result - y_fake_) ** 2)

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        # train generator G
        G.zero_grad()

        # z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        loss_gan = 0
        for i in range(1):
            c_ = Variable(torch.randn((mini_batch, 60)).view(-1, 60, 1, 1).cuda(gpu_id))
            s_ = Variable(torch.randn((mini_batch, 40)).view(-1, 40, 1, 1).cuda(gpu_id))
            G_result = G(c_, s_)
            D_result = D(G_result).squeeze()

            loss_gan += 0.5 * torch.mean((D_result - y_real_) ** 2)

        c2_ = Variable(torch.randn((mini_batch, 60)).view(-1, 60, 1, 1).cuda(gpu_id))
        s2_ = Variable(torch.randn((mini_batch, 40)).view(-1, 40, 1, 1).cuda(gpu_id))

        G_result2s = G(c2_, s_)
        G_result2c = G(c_, s2_)

        A_vgg, A_vgg2c, A_vgg2s = vgg(G_result), vgg(G_result2c), vgg(G_result2s)
        loss_content = weight_c * (criterionVGG(A_vgg, A_vgg2c) + torch.clamp(margin_c-criterionVGG(A_vgg, A_vgg2s), min=0.0))
        if style_loss:
            loss_style = weight_s * (criterionStyle(A_vgg, A_vgg2s) + torch.clamp(margin_s-criterionStyle(A_vgg, A_vgg2c), min=0.0))

        G_train_loss = loss_gan + loss_content + loss_style
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

        if not num_iter % report_period:
            print '# {} - ep: {} - gan: {} - style: {} - content: {} - dis: {}'.format(num_iter, epoch, loss_gan.data.cpu().numpy(), loss_style.data.cpu().numpy(), loss_content.data.cpu().numpy(), D_train_loss.data.cpu().numpy())

        num_iter += 1

        if not num_iter % save_period:
            p = '{}_MYGAN_results/Random_results/{}_MYGAN_'.format(data_name, data_name) + str(epoch + 1) + '.png'
            fixed_p = '{}_MYGAN_results/Fixed_results/{}_MYGAN_'.format(data_name, data_name) + str(epoch + 1) + '.png'
            show_result(G, (epoch + 1), save=True, path=p, isFix=False, fixed_z_=fixed_z_, gpu_id=gpu_id)
            show_result(G, (epoch + 1), save=True, path=fixed_p, isFix=True, fixed_z_=fixed_z_, gpu_id=gpu_id)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    p = '{}_MYGAN_results/Random_results/{}_MYGAN_'.format(data_name, data_name) + str(epoch + 1) + '.png'
    fixed_p = '{}_MYGAN_results/Fixed_results/{}_MYGAN_'.format(data_name, data_name) + str(epoch + 1) + '.png'
    show_result(G, (epoch+1), save=True, path=p, isFix=False, fixed_z_=fixed_z_, gpu_id=gpu_id)
    show_result(G, (epoch+1), save=True, path=fixed_p, isFix=True, fixed_z_=fixed_z_, gpu_id=gpu_id)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    torch.save(G.state_dict(), "{}_MYGAN_results/generator_param.pkl".format(data_name))
    torch.save(D.state_dict(), "{}_MYGAN_results/discriminator_param.pkl".format(data_name))

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "{}_MYGAN_results/generator_param.pkl".format(data_name))
torch.save(D.state_dict(), "{}_MYGAN_results/discriminator_param.pkl".format(data_name))
with open('{}_MYGAN_results/train_hist.pkl'.format(data_name), 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='{}_MYGAN_results/{}_MYGAN_train_hist.png'.format(data_name, data_name))

images = []
for e in range(train_epoch):
    img_name = '{}_MYGAN_results/Fixed_results/{}_MYGAN_'.format(data_name, data_name) + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('{}_MYGAN_results/generation_animation.gif'.format(data_name), images, fps=5)
