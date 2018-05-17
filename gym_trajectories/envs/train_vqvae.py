import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from vqvae import AutoEncoder, to_scalar
#from vqvae_small import AutoEncoder, to_scalar
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time
from glob import glob
import os
from imageio import imread, imwrite
from PIL import Image
from utils import discretized_mix_logistic_loss
from utils import sample_from_discretized_mix_logistic
from utils import get_cuts
from datasets import FroggerDataset


def train(epoch,train_loader,do_use_cuda):
    print("starting epoch {}".format(epoch))
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        if do_use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)
        opt.zero_grad()
        x_d, z_e_x, z_q_x, latents = vmodel(x)
        # with bigger model - latents is 64, 6, 6
        z_q_x.retain_grad()
        #loss_1 = F.binary_cross_entropy(x_d, x)
        # going into dml - x should be bt 0 and 1
        loss_1 = discretized_mix_logistic_loss(x_d,2*x-1,use_cuda=do_use_cuda)
        loss_1.backward(retain_graph=True)
        vmodel.embedding.zero_grad()
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
        loss_2.backward(retain_graph=True)
        loss_3 = .25*F.mse_loss(z_e_x, z_q_x.detach())
        loss_3.backward()
        opt.step()
        train_loss.append(to_scalar([loss_1, loss_2, loss_3]))
        if not batch_idx%10:
            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / float(len(train_loader)),
                np.asarray(train_loss).mean(0),
                time.time() - start_time
            )

    return np.asarray(train_loss).mean(0)

def test(epoch,test_loader,do_use_cuda,save_img_path=None):
    test_loss = []
    for batch_idx, (data, _) in enumerate(test_loader):
        start_time = time.time()
        if do_use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)

        x_d, z_e_x, z_q_x, latents = vmodel(x)
        loss_1 = discretized_mix_logistic_loss(x_d,2*x-1,use_cuda=do_use_cuda)
        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
        loss_3 = .25*F.mse_loss(z_e_x, z_q_x.detach())
        test_loss.append(to_scalar([loss_1, loss_2, loss_3]))
    test_loss_mean = np.asarray(test_loss).mean(0)
    if save_img_path is not None:
        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
        idx = 0
        x_cat = torch.cat([x[idx], x_tilde[idx]], 0)
        images = x_cat.cpu().data
        pred = (((np.array(x_tilde.cpu().data)[0,0]+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
        # input x is between 0 and 1
        real = (np.array(x.cpu().data)[0,0]*float(max_pixel-min_pixel))+min_pixel
        f, ax = plt.subplots(1,3, figsize=(10,3))
        ax[0].imshow(real, vmin=0, vmax=max_pixel)
        ax[0].set_title("original")
        ax[1].imshow(pred, vmin=0, vmax=max_pixel)
        ax[1].set_title("pred epoch %s test loss %s" %(epoch,np.mean(test_loss_mean)))
        ax[2].imshow((pred-real)**2, cmap='gray')
        ax[2].set_title("error")
        f.tight_layout()
        plt.savefig(save_img_path)
        plt.close()
        print("saving example image")
        print("rsync -avhp jhansen@erehwon.cim.mcgill.ca://%s" %os.path.abspath(save_img_path))

    return test_loss_mean


def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


def generate_episodic_npz(data_loader,do_use_cuda,save_path,make_imgs=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for batch_idx, (data, fpaths) in enumerate(data_loader):
        # batch idx must be exactly one episode
        #assert np.sum([fpaths[0][:-10] == f[:-10]  for f in fpaths]) == len(fpaths)
        episode_name = os.path.split(fpaths[0])[1].replace('_frame_00000.png', '.npz')
        episode_path = os.path.join(save_path,episode_name)
        if not os.path.exists(episode_path):
            print("episode: %s" %episode_name)
            start_time = time.time()
            if do_use_cuda:
                x = Variable(data, requires_grad=False).cuda()
            else:
                x = Variable(data, requires_grad=False)

            cuts = get_cuts(x.shape[0], 34)
            for c, (s,e) in enumerate(cuts):
                # make batch
                x_d, z_e_x, z_q_x, latents = vmodel(x[s:e])
                if not c:
                    frame_nums = np.arange(s,e)[:,None]
                    xds = x_d.cpu().data.numpy()
                    zes = z_e_x.cpu().data.numpy()
                    zqs = z_q_x.cpu().data.numpy()
                    ls = latents.cpu().data.numpy()
                else:
                    frame_nums = np.vstack((frame_nums, np.arange(s,e)[:,None]))
                    xds = np.vstack((xds, x_d.cpu().data.numpy()))
                    zes = np.vstack((zes, z_e_x.cpu().data.numpy()))
                    zqs = np.vstack((zqs, z_q_x.cpu().data.numpy()))
                    ls = np.vstack((ls, latents.cpu().data.numpy()))

            # split episode into chunks that are reasonable
            np.savez(episode_path,
                                   z_e_x=zes.astype(np.float32), z_q_x=zqs.astype(np.float32), latents=ls.astype(np.int))







#def generate_results(model,data_loader,nr_logistic_mix,do_use_cuda):
#    start_time = time.time()
#    for batch_idx, (data, img_names) in enumerate(data_loader):
#        if do_use_cuda:
#            x = Variable(data, requires_grad=False).cuda()
#        else:
#            x = Variable(data, requires_grad=False)
#
#        x_d, z_e_x, z_q_x, latents = model(x)
#        # z_e_x is output of encoder
#        # z_q_x is input into decoder
#        # latents is code book
#        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
#        nx_tilde = x_tilde.cpu().data.numpy()
#        nx_tilde = (0.5*nx_tilde+0.5)*255
#        nx_tilde = nx_tilde.astype(np.uint8)
#        embed()
#        #vae_input = z_e_x.contiguous().view(z_e_x.shape[0],-1)
#        #vqvae_rec_images = x_tilde.cpu().data.numpy()
#        nz_q_x = z_q_x.contiguous().view(z_q_x.shape[0],-1).cpu().data.numpy()
#        nlatents = latents.cpu().data.numpy()
#        for ind, img_name in enumerate(img_names):
#            #gen_img_name = img_name.replace('.png', 'vqvae_gen.png')
#            #imwrite(gen_img_name, nx_tilde[ind][0])
#            #latents_name = img_name.replace('.png', 'vqvae_latents.npy')
#            z_q_x_name = img_name.replace('.png', 'vqvae_z_q_x.npy')
#            np.save(z_q_x_name, nz_q_x[ind])
#        if not batch_idx%10:
#            print 'Generate batch_idx: {} Time: {}'.format(
#                batch_idx, time.time() - start_time
#            )

if __name__ == '__main__':
    import argparse
    default_base_datadir = '../../../trajectories_frames/dataset/'
    default_base_savedir = '../../../trajectories_frames/saved/vqvae'

    default_dataset = 'aimgs_48x48'
    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-ds', '--dataset', default='')
    parser.add_argument('-s', '--model_savename', default='base')
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-se', '--save_every', default=5, type=int)
    parser.add_argument('-z', '--num_z', default=32, type=int)
    parser.add_argument('-k', '--num_k', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', default=350, type=int)
    parser.add_argument('-p', '--port', default=8097, type=int, help='8097 for erehwon 8096 for numenor by default')
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-g', '--generate_results', action='store_true', default=False, help='generate dataset of codes')

    args = parser.parse_args()
    train_data_dir = os.path.join(args.datadir, 'train_'+args.dataset+default_dataset)
    test_data_dir =  os.path.join(args.datadir, 'test_'+args.dataset+default_dataset)
    use_cuda = args.cuda

    nr_logistic_mix = 10
    num_clusters = args.num_k
    if use_cuda:
        print("using gpu")
        vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix,num_clusters=num_clusters, encoder_output_size=args.num_z).cuda()
    else:
        vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix,num_clusters=num_clusters, encoder_output_size=args.num_z)

    opt = torch.optim.Adam(vmodel.parameters(), lr=1e-3)
    train_loss_list = []
    test_loss_list = []
    epochs = []
    epoch = 1

    basename = '%s_%s_k%s_z%s_ds%s'%(vmodel.name, args.model_savename,
                                        args.num_k, args.num_z, args.dataset)
    port = args.port
    train_loss_logger = VisdomPlotLogger(
                  'line', port=port, opts={'title': '%s /n Train Loss'%basename})

    test_loss_logger = VisdomPlotLogger(
                  'line', port=port, opts={'title': '%s /n Test Loss'%basename})


    if args.model_loadname is not None:
        model_loadpath = os.path.abspath(os.path.join(default_base_savedir, args.model_loadname))
        if os.path.exists(model_loadpath):
            model_dict = torch.load(model_loadpath)
            vmodel.load_state_dict(model_dict['state_dict'])
            opt.load_state_dict(model_dict['optimizer'])
            epochs.extend(model_dict['epochs'])
            train_loss_list.extend(model_dict['train_losses'])
            test_loss_list.extend(model_dict['test_losses'])
            for e, tr, te in zip(epochs, train_loss_list, test_loss_list):
                train_loss_logger.log(e, np.sum(tr))
                test_loss_logger.log(e, np.sum(te))
            epoch = epochs[-1]+1
            print('loaded checkpoint at epoch: {} from {}'.format(epoch, model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(model_loadpath))
            embed()
    else:
        print('created new model')


    if not args.generate_results:
        data_train_loader = DataLoader(FroggerDataset(train_data_dir,
                                       transform=transforms.ToTensor(), limit=args.num_train_limit),
                                       batch_size=32, shuffle=True)
        data_test_loader = DataLoader(FroggerDataset(test_data_dir,
                                      transform=transforms.ToTensor()),
                                      batch_size=32, shuffle=True)


        #    test_img = args.model_savepath.replace('.pkl', '_test.png')
        for e in xrange(epoch,epoch+args.num_epochs):
            train_loss = train(e,data_train_loader,do_use_cuda=use_cuda)
            if (not e%args.save_every) or (e==epoch+args.num_epochs):
                print('------------------------------------------------------------')
                test_img_name = os.path.join(default_base_savedir , basename + "e%05d.png"%e)
                filename = os.path.join(default_base_savedir , basename + "e%05d.pkl"%e)
                test_loss = test(e,data_test_loader,do_use_cuda=use_cuda,save_img_path=test_img_name)

                epochs.append(e)
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                print('send data to plotter')

                train_loss_logger.log(e, np.sum(train_loss_list[-1]))
                test_loss_logger.log(e,  np.sum(test_loss_list[-1]))
                print('train_loss {} test_loss {}'.format(train_loss,test_loss))
                print('train_loss sum {} test_loss sum {}'.format(np.sum(train_loss),np.sum(test_loss)))
                state = {'epoch':e,
                         'epochs':epochs,
                         'state_dict':vmodel.state_dict(),
                         'train_losses':train_loss_list,
                         'test_losses':test_loss_list,
                         'optimizer':opt.state_dict(),
                         }

                save_checkpoint(state, filename=filename)
    else:
        if args.model_loadname is None:
            print("must give valid model!")
            sys.exit()
        episode_length = 203
        data_train_loader = DataLoader(FroggerDataset(train_data_dir,
                                       transform=transforms.ToTensor(), limit=args.num_train_limit),
                                       batch_size=episode_length, shuffle=False)
        data_test_loader = DataLoader(FroggerDataset(test_data_dir,
                                      transform=transforms.ToTensor()),
                                      batch_size=episode_length, shuffle=False)


        test_gen_dir = os.path.join(args.datadir, 'test_'  + basename+'_e%05d'%epoch)
        train_gen_dir = os.path.join(args.datadir, 'train_'+ basename+'_e%05d'%epoch)

        generate_episodic_npz(data_test_loader,use_cuda,test_gen_dir)
        generate_episodic_npz(data_train_loader,use_cuda,train_gen_dir)



