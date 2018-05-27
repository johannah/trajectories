from torchnet.logger import VisdomPlotLogger, VisdomLogger
import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from conv_vae import Encoder, Decoder, VAE, latent_loss
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time
from glob import glob
import os
from imageio import imread, imwrite
from PIL import Image
from utils import discretized_mix_logistic_loss
from utils import sample_from_discretized_mix_logistic, to_scalar
from datasets import FroggerDataset
from datasets import max_pixel, min_pixel

def train(epoch,train_loader,DEVICE):
    print("starting epoch {}".format(epoch))
    train_loss = []
    kl_weight = 1
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        x = Variable(data, requires_grad=False).to(DEVICE)
        opt.zero_grad()
        x_di = vmodel(x)
        # use cuda?
        dmll_loss = discretized_mix_logistic_loss(x_di, 2*x-1, nr_mix=nr_mix,use_cuda=args.cuda)
        kl_loss = kl_weight*latent_loss(vmodel.z_mean, vmodel.z_sigma)
        loss = dmll_loss+kl_loss
        loss.backward()
        opt.step()
        train_loss.append(to_scalar([kl_loss, dmll_loss]))

        if not batch_idx%10:
            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / float(len(train_loader)),
                np.asarray(train_loss).mean(0),
                time.time() - start_time
            )

    return np.asarray(train_loss).mean(0)

def test(epoch,test_loader,DEVICE,save_img_path=None):
    test_loss = []
    kl_weight = 1
    for batch_idx, (data, _) in enumerate(test_loader):
        start_time = time.time()
        x = Variable(data, requires_grad=False).to(DEVICE)
        x_di = vmodel(x)
        # use cuda?
        dmll_loss = discretized_mix_logistic_loss(x_di, 2*x-1, nr_mix=nr_mix,use_cuda=args.cuda)
        kl_loss = kl_weight*latent_loss(vmodel.z_mean, vmodel.z_sigma)
        loss = dmll_loss+kl_loss
        test_loss.append(to_scalar([kl_loss, dmll_loss]))

    test_loss_mean = np.asarray(test_loss).mean(0)
    #if save_img_path is not None:
    #    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
    #    idx = 0
    #    x_cat = torch.cat([x[idx], x_tilde[idx]], 0)
    #    images = x_cat.cpu().data
    #    pred = (((np.array(x_tilde.cpu().data)[0,0]+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
    #    # input x is between 0 and 1
    #    real = (np.array(x.cpu().data)[0,0]*float(max_pixel-min_pixel))+min_pixel
    #    f, ax = plt.subplots(1,3, figsize=(10,3))
    #    ax[0].imshow(real, vmin=0, vmax=max_pixel)
    #    ax[0].set_title("original")
    #    ax[1].imshow(pred, vmin=0, vmax=max_pixel)
    #    ax[1].set_title("pred epoch %s test loss %s" %(epoch,np.mean(test_loss_mean)))
    #    ax[2].imshow((pred-real)**2, cmap='gray')
    #    ax[2].set_title("error")
    #    f.tight_layout()
    #    plt.savefig(save_img_path)
    #    plt.close()
    #    print("saving example image")
    #    print("rsync -avhp jhansen@erehwon.cim.mcgill.ca://%s" %os.path.abspath(save_img_path))

    return test_loss_mean


def save_checkpoint(state, filename='model.pkl'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)


def generate_episodic_npz(data_loader,DEVICE,save_path,make_imgs=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for batch_idx, (data, fpaths) in enumerate(data_loader):
        # batch idx must be exactly one episode
        #assert np.sum([fpaths[0][:-10] == f[:-10]  for f in fpaths]) == len(fpaths)

        episode_name = os.path.split(fpaths[0])[1].replace('_frame_00000.png', '.npz')
        episode_path = os.path.join(save_path,episode_name)
        if not os.path.exists(episode_path):
            print("episode: %s" %episode_name)
            if not 'npz' in episode_name:
                embed()
            start_time = time.time()
            x = Variable(data, requires_grad=False).to(DEVICE)

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
            np.savez(episode_path, z_e_x=zes.astype(np.float32), z_q_x=zqs.astype(np.float32), latents=ls.astype(np.int))


if __name__ == '__main__':
    import argparse
    default_base_datadir = '../../../trajectories_frames/dataset/'
    default_base_savedir = '../../../trajectories_frames/saved/vae'

    default_dataset = 'imgs_48x48'
    parser = argparse.ArgumentParser(description='train vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-ds', '--dataset', default='')
    parser.add_argument('-s', '--model_savename', default='base')
    parser.add_argument('-l', '--model_loadname', default=None)
    parser.add_argument('-se', '--save_every', default=5, type=int)
    parser.add_argument('-z', '--num_z', default=32, type=int)
    parser.add_argument('-k', '--num_k', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', default=350, type=int)
    parser.add_argument('-p', '--port', default=8096, type=int, help='8097 for erehwon 8096 for numenor by default')
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-g', '--generate_results', action='store_true', default=False, help='generate dataset of codes')

    args = parser.parse_args()
    train_data_dir = os.path.join(args.datadir, 'train_'+args.dataset+default_dataset)
    test_data_dir =  os.path.join(args.datadir, 'test_'+args.dataset+default_dataset)
    use_cuda = args.cuda

    if use_cuda:
        DEVICE = 'cuda'
        print("using gpu")
    else:
        DEVICE = 'cpu'

    dsize = 48
    # mean and scale for each components and weighting bt components (10+2*10)
    nr_mix = 10
    probs_size = (2*nr_mix)+nr_mix
    latent_size = 32

    # mu_size is 800 for 40x40
    # mu_size is 1152 for 48x48
    mu_size = 1152
    encoder = Encoder(latent_size).to(DEVICE)
    decoder = Decoder(latent_size, probs_size).to(DEVICE)
    vmodel = VAE(encoder, decoder, mu_size=mu_size, device=DEVICE).to(DEVICE)


    opt = torch.optim.Adam(vmodel.parameters(), lr=1e-4)
    train_loss_list = []
    test_loss_list = []
    epochs = []
    epoch = 1

    basename = '%s_%s_k%s_z%s_ds%s'%(vmodel.name, args.model_savename,
                                        args.num_k, args.num_z, args.dataset)
    port = args.port
    train_loss_logger = VisdomPlotLogger(
                  'line', port=port, opts={'title': '%s Train'%basename})

    test_loss_logger = VisdomPlotLogger(
                  'line', port=port, opts={'title': '%s  Test'%basename})


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

    embed()
    if not args.generate_results:
        print("starting data loader")
        data_train_loader = DataLoader(FroggerDataset(train_data_dir,
                                       transform=transforms.ToTensor(), limit=args.num_train_limit),
                                       batch_size=32, num_workers=4, shuffle=True)
        data_test_loader = DataLoader(FroggerDataset(test_data_dir,
                                      transform=transforms.ToTensor()),
                                      batch_size=32, num_workers=2, shuffle=True)


        #    test_img = args.model_savepath.replace('.pkl', '_test.png')
        for e in xrange(epoch,epoch+args.num_epochs):
            train_loss = train(e,data_train_loader,DEVICE=DEVICE)
            print('------------------------------------------------------------')
            test_img_name = os.path.join(default_base_savedir , basename + "e%05d.png"%e)
            filename = os.path.join(default_base_savedir , basename + "e%05d.pkl"%e)

            test_loss = test(e,data_test_loader,DEVICE=DEVICE,save_img_path=test_img_name)

            epochs.append(e)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            print('send data to plotter')

            train_loss_logger.log(e, np.sum(train_loss_list[-1]))
            test_loss_logger.log(e,  np.sum(test_loss_list[-1]))
            print('train_loss {} test_loss {}'.format(train_loss,test_loss))
            print('train_loss sum {} test_loss sum {}'.format(np.sum(train_loss),np.sum(test_loss)))

            if (not e%args.save_every) or (e==epoch+args.num_epochs):
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







####################################################################
#def train(epoch,model,optimizer,train_loader,do_checkpoint,do_use_cuda):
#    latent_losses = []
#    dmll_losses = []
#    kl_weight = 1.0
#    for batch_idx, (data, _) in enumerate(train_loader):
#        start_time = time.time()
#        if do_use_cuda:
#            x = Variable(data, requires_grad=False).cuda()
#        else:
#            x = Variable(data, requires_grad=False)
#
#        optimizer.zero_grad()
#        x_di = vae(x)
#        dmll_loss = discretized_mix_logistic_loss(x_di, 2*x-1, nr_mix=nr_mix, use_cuda=do_use_cuda)
#        kl_loss = kl_weight*latent_loss(vae.z_mean, vae.z_sigma)
#        loss = dmll_loss+kl_loss
#        loss.backward()
#        optimizer.step()
#        latent_losses.append(kl_loss.cpu().data.numpy().mean())
#        dmll_losses.append(dmll_loss.cpu().data.numpy().mean())
#
#        if not batch_idx%100:
#            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tKL Loss: {} X Loss: {} Time: {}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / float(len(train_loader)),
#                np.asarray(latent_losses).mean(0),
#                np.asarray(dmll_losses).mean(0),
#                time.time() - start_time
#            )
#
#    state = {'epoch':epoch,
#             'state_dict':vae.state_dict(),
#             'dmll_losses':np.asarray(dmll_losses).mean(0),
#             'latent_losses':np.asarray(latent_losses).mean(0),
#             'optimizer':optimizer.state_dict(),
#             }
#    return model, optimizer, state
#
#
#def save_checkpoint(state, is_best=False, filename='model.pkl'):
#    torch.save(state, filename)
#    if is_best:
#        bestpath = os.path.join(os.path.split(filename)[0], 'model_best.pkl')
#        shutil.copyfile(filename, bestpath)
#
#def generate_results(base_path,data_loader,nr_logistic_mix,do_use_cuda):
#    start_time = time.time()
#    data_mu = np.empty((0,800))
#    data_sigma = np.empty((0,800))
#    limit = 4000
#    for batch_idx, (data, img_names) in enumerate(data_loader):
#        if batch_idx*32 > limit:
#            continue
#        else:
#            if do_use_cuda:
#                x = Variable(data, requires_grad=False).cuda()
#            else:
#                x = Variable(data, requires_grad=False)
#
#            x_d = vae(x)
#            x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
#            nx_tilde = x_tilde.cpu().data.numpy()
#            inx_tilde = ((0.5*nx_tilde+0.5)*255).astype(np.uint8)
#            # vae.z_mean is batch_sizex800
#            # vae.z_sigma is batch_sizex800
#            zmean = vae.z_mean.cpu().data.numpy()
#            zsigma = vae.z_sigma.cpu().data.numpy()
#            data_mu = np.vstack((data_mu,zmean))
#            data_sigma = np.vstack((data_sigma,zsigma))
#            for ind, img_path in enumerate(img_names):
#                img_name = os.path.split(img_path)[1]
#                gen_img_name = img_name.replace('.png', 'conv_vae_gen.png')
#                #gen_latent_name = img_name.replace('.png', 'conv_vae_latents.npz')
#                imwrite(os.path.join(base_path,gen_img_name), inx_tilde[ind][0])
#                #np.savez(gen_latent_name, zmean=zmean[ind], zsigma=zsigma[ind])
#            if not batch_idx%10:
#                print 'Generate batch_idx: {} Time: {}'.format(
#                    batch_idx, time.time() - start_time
#                )
#    np.savez(os.path.join(base_path,'mu_conv_vae.npz'), data_mu)
#    np.savez(os.path.join(base_path,'sigma_conv_vae.npz'), data_sigma)
#
#if __name__ == '__main__':
#    import argparse
#    default_base_datadir = '/localdata/jhansen/trajectories_frames/saved/vae'
#    default_model_savepath = os.path.join(default_base_datadir, 'conv_vae_model.pkl')
#
#    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
#    parser.add_argument('-c', '--cuda', action='store_true', default=False)
#    parser.add_argument('-d', '--datadir', default=default_base_datadir)
#    parser.add_argument('-s', '--model_savepath', default=default_model_savepath)
#    parser.add_argument('-l', '--model_loadpath', default=None)
#    parser.add_argument('-z', '--num_z', default=32, type=int)
#    parser.add_argument('-e', '--num_epochs', default=350, type=int)
#    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
#    parser.add_argument('-g', '--generate_results', action='store_true', default=False, help='generate dataset of codes')
#
#    args = parser.parse_args()
#    train_data_dir = os.path.join(args.datadir, 'imgs_train')
#    test_data_dir =  os.path.join(args.datadir, 'imgs_test')
#    use_cuda = args.cuda
#
#    dsize = 48
#    nr_mix = 10
#    # mean and scale for each components and weighting bt components (10+2*10)
#    probs_size = (2*nr_mix)+nr_mix
#    latent_size = 32
#
#    encoder = Encoder(latent_size)
#    decoder = Decoder(latent_size, probs_size)
#    vae = VAE(encoder, decoder, use_cuda)
#    # square error is not the correct loss - for ordered input,
#    # should use softmax for unordered input ( like mine )
#
#    if use_cuda:
#        print("using gpu")
#        vae = vae.cuda()
#        vae.encoder = vae.encoder.cuda()
#        vae.decoder = vae.decoder.cuda()
#    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
#    epoch = 0
#    data_train_loader = DataLoader(FroggerDataset(train_data_dir,
#                                   transform=transforms.ToTensor(),
#                                   limit=args.num_train_limit),
#                                   batch_size=64, shuffle=True)
#    data_test_loader = DataLoader(FroggerDataset(test_data_dir,
#                                   transform=transforms.ToTensor()),
#                                  batch_size=32, shuffle=True)
#    test_data = data_test_loader
#
#    if args.model_loadpath is not None:
#        if os.path.exists(args.model_loadpath):
#            model_dict = torch.load(args.model_loadpath)
#            vae.load_state_dict(model_dict['state_dict'])
#            opt.load_state_dict(model_dict['optimizer'])
#            epoch =  model_dict['epoch']
#            print('loaded checkpoint at epoch: {} from {}'.format(epoch, args.model_loadpath))
#        else:
#            print('could not find checkpoint at {}'.format(args.model_loadpath))
#            embed()
#
#    for batch_idx, (test_data, _) in enumerate(data_test_loader):
#        if use_cuda:
#            x_test = Variable(test_data).cuda()
#        else:
#            x_test = Variable(test_data)
#
#    if not args.generate_results:
#        for e in xrange(epoch,epoch+args.num_epochs):
#            vae, opt, state = train(e,vae,opt,data_train_loader,
#                                do_checkpoint=True,do_use_cuda=use_cuda)
#            #test_loss = test(x_test,vae,do_use_cuda=use_cuda,save_img_path=test_img)
#            #print('test_loss {}'.format(test_loss))
#            #state['test_loss'] = test_loss
#            save_checkpoint(state, filename=args.model_savepath)
#
#    else:
#        #generate_results('../saved/test_results/', data_test_loader,nr_mix,use_cuda)
#        generate_results('../saved/train_results/', data_train_loader,nr_mix,use_cuda)
#







