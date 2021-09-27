# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:27:37 2021

@author: chris
"""

from VAE_arch3 import *
from load_data_new import load_data_new
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
from mpl_toolkits import mplot3d

plt.close('all')

def load_VAE_arch3(path):
    config = torch.load(path)
    data_channels = 3
    initial_features = config['initial_features']
    growth_rate = config['growth_rate']
    n_latent = config['n_latent']
    VAE = VAE_arch3(data_channels, initial_features, growth_rate, n_latent)
    VAE.load_state_dict(config['model_state_dict'])
    loss_reg = config['l_reg']
    loss_rec = config['l_rec']
    beta_list = config['beta_final']
    #beta_list = 0
    return VAE, loss_reg, loss_rec, beta_list, config

n_latent = 2
kle = 2
ntrain = 512
ntest = 512

arch_num = 3
trial = 3
load_path = './DarcyFlow/p2/multimodal/arch{}/n2/VAE_{}'.format(arch_num,trial)
model_name = 'VAE_{}.pth'.format(trial)

save_figs = True

train_data_dir = 'data/DarcyFlow/multimodal/kle{}_lhs{}_bimodal_2.hdf5'.format(kle, ntrain)#
test_data_dir = 'data/DarcyFlow/multimodal/kle{}_mc{}_bimodal_2.hdf5'.format(kle, ntest)#

train_loader, train_stats = load_data_new(train_data_dir, ntrain)
test_loader, test_stats = load_data_new(test_data_dir, ntest)

VAE, loss_reg, loss_rec, beta, config = load_VAE_arch3(os.path.join(load_path, model_name))#'gauss_location_only_big_condensed_VAEs/DenseVAE_n2_1.pth')#

print('Loss: ', loss_reg + loss_rec)
    
for n, (g, perm, output) in enumerate(train_loader):
    if n == 0:
        zmu, zlogvar = VAE.encoder(output)
        plt.figure(1)
        plt.plot(zmu[:,0].detach().numpy(), zmu[:,1].detach().numpy(), '.')
        plt.figure(2)
        plt.plot(zlogvar[:,0].detach().numpy(), zlogvar[:,1].detach().numpy(), '.')
        in_test = output
        zmu, _, z, out_test, out_test_logvar = VAE.forward(output)

        out1mu = VAE.dec1(torch.unsqueeze(z[:,0],-1))
        out2mu = VAE.dec2(torch.unsqueeze(z[:,1],-1))
        
        plt.figure(3)
        plt.subplot(2,2,1)
        plt.imshow(output[0,1,:,:], cmap = 'jet')
        plt.colorbar()
        plt.subplot(2,2,2)
        plt.imshow(out_test[0,1,:,:].detach().numpy(), cmap = 'jet')
        plt.colorbar()
        
        #plt.subplot(2,2,4)
        #plt.imshow(np.exp(out_test_logvar[0,1,:,:].detach().numpy()), cmap = 'jet')
        #plt.colorbar()
        ind = 5
        plt.figure(126)
        plt.subplot(2,3,1)
        plt.imshow(out1mu[ind,1,:,:].detach().numpy(), cmap = 'jet')
        plt.subplot(2,3,2)
        plt.imshow(out2mu[ind,1,:,:].detach().numpy(), cmap = 'jet')
        plt.subplot(2,3,3)
        plt.imshow(out1mu[ind,0,:,:].detach().numpy(), cmap = 'jet')
        plt.subplot(2,3,4)
        plt.imshow(out2mu[ind,2,:,:].detach().numpy(), cmap = 'jet')
        plt.subplot(2,3,5)
        plt.imshow(out1mu[ind,2,:,:].detach().numpy(), cmap = 'jet')
        plt.subplot(2,3,6)
        plt.imshow(out2mu[ind,0,:,:].detach().numpy(), cmap = 'jet')
        
for n, (g_test, _, output) in enumerate(test_loader):
    if n == 0:
        #in_test = output
        zmu_test, zlogvar_test, z_test, out_test_test, _ = VAE.forward(output)

print('z shape: ', np.shape(zmu))
plt.figure(23)
plt.subplot(2,2,1)
plt.imshow(perm[0,0,:,:], cmap = 'jet')
plt.subplot(2,2,2)
plt.imshow(output[0,0,:,:], cmap='jet')
plt.subplot(2,2,3)
plt.imshow(output[0,1,:,:], cmap = 'jet')
plt.subplot(2,2,4)
plt.imshow(output[0,2,:,:], cmap='jet')

plt.figure(43)
plt.plot(beta, lw=3)
plt.xlabel('Epoch', fontsize = 22)
plt.ylabel(r'$\beta$', fontsize = 22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
if save_figs:
    plt.savefig(os.path.join(load_path, 'beta_{}.png'.format(trial)))

in_test = in_test.detach().numpy()
out_test = out_test.detach().numpy()
x = np.linspace(0, 1, 65)
y = np.linspace(0, 1, 65)
[X,Y] = np.meshgrid(x,y)

in_test = in_test[0,1,:,:]
out_test = out_test[0,1,:,:]
plt.figure(4, figsize = (14.6, 3.5))
plt.subplot(1,3,1)
plt.imshow(in_test, cmap = 'jet')
plt.title('Data Sample', fontsize = 16)
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(out_test, cmap = 'jet')
plt.title('Reconstructed Mean', fontsize = 16)
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(in_test-out_test, cmap = 'jet')
plt.title('Error in Mean', fontsize = 16)
plt.colorbar()
if save_figs:
    plt.savefig(os.path.join(load_path, 'recon_{}.png'.format(trial)))

plt.figure(10)
plt.plot(beta*loss_reg + loss_rec, 'r', label = r'Training Loss', lw=3)
plt.plot(loss_reg + loss_rec, 'k', label = r'$L_{VAE}$', lw=3)
plt.legend(prop={"size":16})
plt.ylabel(r'Loss', fontsize=22)
plt.xlabel(r'Epochs', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
if save_figs:
    plt.savefig(os.path.join(load_path, 'training_losses_{}.png'.format(trial)))

plt.figure(13)
ind = config['epochs'] - 1250

# plot all latent data correlations with gen parameters
plt.figure(6, figsize = (13, 13))
count = 0
for i in range(n_latent):
    for j in range(kle):
        if j == 0:
            count = count + 2
        else:
            count = count + 1
        plt.subplot(n_latent+1, kle+1, count)
        if (i == 0) & (j==kle-1):
            plt.plot(g[:,j], zmu[:,i].detach().numpy(), 'k.', markersize = 1, label = 'Train Data')
            plt.plot(g_test[:,j],zmu_test[:,i].detach().numpy(),'r.', markersize = 1, label = 'Test Data')
            plt.legend(prop={"size":14}, markerscale = 3.)
        else:
            plt.plot(g[:,j], zmu[:,i].detach().numpy(), 'k.', markersize = 1)
            plt.plot(g_test[:,j],zmu_test[:,i].detach().numpy(),'r.', markersize = 1)
            
ztest = zmu.detach().numpy()

kde = KernelDensity(bandwidth = 0.5, kernel = 'gaussian')
for i in range(n_latent):
    v = np.concatenate((g[:,i], g_test[:,i]), axis = 0)
    kde.fit(np.reshape(v, (-1,1)))
    plotv = np.linspace(np.min(v), np.max(v), 1000)
    lp = kde.score_samples(np.reshape(plotv, (-1, 1)))
    plt.subplot(n_latent+1, kle+1, (n_latent*(kle+1) + i + 2))
    plt.xlabel(r'$\theta_{}$'.format(i+1), fontsize = 18)
    plt.plot(plotv, np.exp(lp), 'k--')
    
kde = KernelDensity(bandwidth = 0.5, kernel = 'gaussian')
for i in range(n_latent):
    v = np.concatenate((z[:,i].detach().numpy(), z_test[:,i].detach().numpy()), axis = 0)
    kde.fit(np.reshape(v, (-1,1)))
    plotv = np.linspace(np.min(v), np.max(v), 1000)
    lp = kde.score_samples(np.reshape(plotv, (-1, 1)))
    plt.subplot(n_latent+1, kle+1, i*(kle+1) + 1)
    plt.xlabel(r'$z_{}$'.format(i+1), fontsize = 18)
    plt.plot( np.exp(lp), plotv, 'k--')
if save_figs:
    plt.savefig(os.path.join(load_path, 'disentanglement_{}.png'.format(trial)))

ztest = zmu.detach().numpy()
ztest = ztest[:,0:2]
g = g[:,0:2]
#ztest = np.reshape(ztest, (-1, 1))
n_samples = 100
xv1 = np.linspace(-4, 4, 100)
xv2 = xv1 + 0.
[XM1, XM2] = np.meshgrid(xv1, xv2)
fitM = np.concatenate((np.reshape(XM1, (-1,1)), np.reshape(XM2, (-1,1))), axis = 1)
kde = KernelDensity(bandwidth = 0.5, kernel = 'gaussian')
kde.fit(ztest)
lp = kde.score_samples(fitM)
kde.fit(g)
lpg = kde.score_samples(fitM)
prior = multivariate_normal([0, 0], [[1, 0], [0, 1]])
plt.figure(7, figsize=(7,7))
plt.contour(XM1, XM2, np.exp(np.reshape(lp, (n_samples, n_samples))), colors = 'red')
plt.contour(XM1, XM2, np.exp(np.reshape(lpg, (n_samples, n_samples))), colors = 'blue')
plt.contour(XM1, XM2, prior.pdf(np.dstack((XM1, XM2))), colors = 'black')
plt.gca().set_aspect('equal') 
plt.xlabel(r'$z_1$', fontsize = 18)
plt.ylabel(r'$z_2$', fontsize = 18)
plt.title('Aggregated Posterior - Prior Comparison (VAE)', fontsize = 16)
proxy = [plt.Rectangle((0,0),1,1,fc = 'red'), plt.Rectangle((0,0),1,1,fc = 'blue'), plt.Rectangle((0,0),1,1,fc = 'black')]
plt.legend(proxy, ['Aggregated Posterior', 'Generative Parameters', 'Prior'], prop={"size":16}, loc = 2)
if save_figs:
    plt.savefig(os.path.join(load_path, 'agg_post_comparison_{}.png'.format(trial)))

plt.figure(27)
plt.plot(zmu[:,0].detach().numpy(), zmu[:,1].detach().numpy(), 'k.', markersize = 1, label = 'Train Data')
plt.plot(zmu_test[:,0].detach().numpy(), zmu_test[:,1].detach().numpy(), 'r.', markersize = 1, label = 'Test Data')
plt.xlabel(r'$z_1$')
plt.ylabel(r'$z_2$')
plt.title(r'Samples from $q_\phi(z)$')
if save_figs:
    plt.savefig(os.path.join(load_path, 'agg_post_samples_{}.png'.format(trial)))
'''

# plot test data correlations with generative parameters (include uncertainty)

plt.figure(8)
plt.subplot(1,2,1)
g1_test_sort = np.sort(g_test[:,0])
inds = np.argsort(g_test[:,0])
zmu1_test_sort = zmu_test[inds,0].detach().numpy()
zmu1_std_sort = np.exp(0.5*zlogvar_test[inds,0].detach().numpy())
plt.fill_between(g1_test_sort, zmu1_test_sort + 2*zmu1_std_sort, zmu1_test_sort - 2*zmu1_std_sort, alpha = 0.15)
plt.plot(g1_test_sort, zmu1_test_sort,'k.', markersize = 1)
plt.xlabel(r'$\theta_1$', fontsize = 18)
plt.ylabel(r'$z_1$', fontsize = 18)
#plt.gca().set_aspect('equal')

plt.subplot(1,2,2)
g2_test_sort = np.sort(g_test[:,1])
inds = np.argsort(g_test[:,1])
zmu2_test_sort = zmu_test[inds,1].detach().numpy()
zmu2_std_sort = np.exp(0.5*zlogvar_test[inds,1].detach().numpy())
plt.fill_between(g2_test_sort, zmu2_test_sort + 2*zmu2_std_sort, zmu2_test_sort - 2*zmu2_std_sort, alpha = 0.15, label = r'$\pm 2\sigma_{z}$')
plt.plot(g2_test_sort, zmu2_test_sort,'k.', markersize = 1, label = r'$\mu_z$')
plt.xlabel(r'$\theta_2$', fontsize = 18)
plt.ylabel(r'$z_2$', fontsize = 18)
plt.legend(prop={"size":14})
#plt.gca().set_aspect('equal')


z = zmu.detach().numpy()
fig = plt.figure(21)
ax = plt.axes(projection='3d')
ax.scatter3D(z[:,0], z[:,1], z[:,2], c='k')

'''

