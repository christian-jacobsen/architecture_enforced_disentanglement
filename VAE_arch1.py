# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:20:17 2021

@author: christian jacobsen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np



class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.

    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    def __init__(self, in_features, growth_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        
    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, n_layers, in_features, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(n_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True, 
                 drop_rate=0):
        """Transition layer, either downsampling or upsampling, both reduce
        number of feature maps, i.e. `out_features` should be less than 
        `in_features`.

        Args:
            in_features (int):
            out_features (int):
            down (bool): If True, downsampling, else upsampling
            bottleneck (bool, True): If True, enable bottleneck design
            drop_rate (float, 0.):
        """
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if down:
            # half feature resolution, reduce # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
                # not using pooling, fully convolutional...
                self.add_module('conv2', nn.Conv2d(out_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
        else:
            # transition up, increase feature resolution, half # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # output_padding=0, or 1 depends on the image size
                # if image size is of the power of 2, then 1 is good
                self.add_module('convT2', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('convT1', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

def last_decoding(in_features, out_channels, kernel_size, stride, padding, 
                  output_padding=0, bias=False, drop_rate=0.):
    """Last transition up layer, which outputs directly the predictions.
    """
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm2d(in_features))
    last_up.add_module('relu1', nn.ReLU(True))
    last_up.add_module('conv1', nn.Conv2d(in_features, in_features // 2, 
                    kernel_size=1, stride=1, padding=0, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout2d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm2d(in_features // 2))
    last_up.add_module('relu2', nn.ReLU(True))
    last_up.add_module('convT2', nn.ConvTranspose2d(in_features // 2, 
                       out_channels, kernel_size=kernel_size, stride=stride, 
                       padding=padding, output_padding=output_padding, bias=bias))
    return last_up


class VAE_arch1(nn.Module):
    def __init__(self, data_channels, initial_features, growth_rate, n_latent, prior = 'std_norm', activations = nn.ReLU()):
        """
        A VAE using convolutional dense blocks and convolutional encoding layers
        """
        
        super(VAE_arch1, self).__init__()
        
        self.data_channels = data_channels
        self.K = growth_rate
        self.n_latent = n_latent
        self.act = activations
        
        
        self.enc1 = nn.Sequential()
        self.enc2 = nn.Sequential()
        self.cast_m1 = nn.Sequential() # output is mean of first latent dim
        self.cast_lv1 = nn.Sequential() # output is logvar of first latent dim
        self.cast_m2 = nn.Sequential()
        self.cast_lv2 = nn.Sequential()
        
        self.dec1 = nn.Sequential()
        self.dec2 = nn.Sequential()  # decode each dim seperately, then comvine
        self.m_out = nn.Sequential()
        self.lv_out = nn.Sequential()
        
        
        n_l = 4
        
        # define the prior log var --- prior assumed gaussian here
        if prior == 'scaled_gaussian':
            self.prior_logvar = torch.Tensor([np.log(1), np.log(2)]) # change the scale of prior
        elif prior == 'std_norm':
            self.prior_logvar = torch.zeros(self.n_latent) # the standard prior
            
        self.prior_logvar = self.prior_logvar.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        
        self.enc1.add_module('Initial_convolution', nn.Conv2d(data_channels, initial_features,
                                                              kernel_size = 7, stride = 2, padding = 2, bias = False))
        
        self.dec_logvar = nn.Parameter(torch.zeros((65, 65)), requires_grad = True)
        
        n_features = initial_features
        
        self.enc1.add_module('EncoderDenseBlock', _DenseBlock(n_layers = n_l, in_features = n_features, growth_rate = self.K))
        n_features = n_features + n_l*self.K
        self.enc1.add_module('Encode', _Transition(in_features = n_features, out_features = n_features // 2, down = True))
        n_features = n_features // 2
        n_features1 = n_features
        
        self.enc2.add_module('EncoderDenseBlock', _DenseBlock(n_layers = n_l, in_features = n_features, growth_rate = self.K))
        n_features = n_features + n_l*self.K
        self.enc2.add_module('Encode', _Transition(in_features = n_features, out_features = n_features // 2, down = True))
        n_features = n_features // 2
        n_features2 = n_features
        
        flatten_dim2 = n_features2*8*8 
        self.cast_m2.add_module('Flatten', nn.Flatten())
        self.cast_m2.add_module('FullConn1', nn.Linear(flatten_dim2, flatten_dim2 // 2))
        self.cast_m2.add_module('Act1', self.act)
        self.cast_m2.add_module('FullConn2', nn.Linear(flatten_dim2 // 2, 1))
        
        self.cast_lv2.add_module('Flatten', nn.Flatten())
        self.cast_lv2.add_module('FullConn1', nn.Linear(flatten_dim2, flatten_dim2 // 2))
        self.cast_lv2.add_module('Act1', self.act)
        self.cast_lv2.add_module('FullConn2', nn.Linear(flatten_dim2 // 2, 1))
        
        flatten_dim1 = n_features1*16*16
        self.cast_m1.add_module('Flatten', nn.Flatten())
        self.cast_m1.add_module('FullConn1', nn.Linear(flatten_dim1, flatten_dim1 // 2))
        self.cast_m1.add_module('Act1', self.act)
        self.cast_m1.add_module('FullConn2', nn.Linear(flatten_dim1 // 2, 1))
        
        self.cast_lv1.add_module('Flatten', nn.Flatten())
        self.cast_lv1.add_module('FullConn1', nn.Linear(flatten_dim1, flatten_dim1 // 2))
        self.cast_lv1.add_module('Act1', self.act)
        self.cast_lv1.add_module('FullConn2', nn.Linear(flatten_dim1 // 2, 1))
        

        self.dec1.add_module('FullConn1', nn.Linear(1, flatten_dim1 // 2))
        self.dec1.add_module('Act1', nn.ReLU())
        self.dec1.add_module('FullConn2', nn.Linear(flatten_dim1 // 2, flatten_dim1))
        self.dec1.add_module('Reshape1', Reshape((-1, n_features1, 16, 16)))
        
        n_features = n_features1
        for i, n_layers in enumerate([n_l]):
            block = _DenseBlock(n_layers = n_layers, in_features = n_features, growth_rate = self.K)
            
            self.dec1.add_module('DecoderDenseBlock%d' % (i+1), block)
            n_features += n_layers*self.K
            
            dec = _Transition(in_features = n_features, out_features = n_features // 2,
                              down = False)
            self.dec1.add_module('DecodeUp%d' % (i+1), dec)
            n_features = n_features // 2
            
        final_decode = last_decoding(n_features, data_channels, kernel_size = 4, stride = 2, padding = 1, 
                         output_padding = 1, bias = False, drop_rate = 0)
                
        self.dec1.add_module('FinalDecode', final_decode)
        #self.dec1.add_module('FinalConv', nn.Conv2d(data_channels, data_channels, 
        #                                                         kernel_size = 5, stride = 1, padding = 2, bias = False))
        
            
        self.dec2.add_module('FullConn1', nn.Linear(1, flatten_dim2 // 2))
        self.dec2.add_module('Act1', nn.ReLU())
        self.dec2.add_module('FullConn2', nn.Linear(flatten_dim2 // 2, flatten_dim2))
        self.dec2.add_module('Reshape1', Reshape((-1, n_features2 , 8, 8)))
        
        n_features = n_features2
        for i, n_layers in enumerate([n_l+2, n_l]):
            block = _DenseBlock(n_layers = n_layers, in_features = n_features, growth_rate = self.K)
            
            self.dec2.add_module('DecoderDenseBlock%d' % (i+1), block)
            n_features += n_layers*self.K
            
            dec = _Transition(in_features = n_features, out_features = n_features // 2,
                              down = False)
            self.dec2.add_module('DecodeUp%d' % (i+1), dec)
            n_features = n_features // 2
            
                            
        final_decode = last_decoding(n_features, data_channels, kernel_size = 4, stride = 2, padding = 1, 
                                     output_padding = 1, bias = False, drop_rate = 0)
                
        self.dec2.add_module('FinalDecode', final_decode)
        #self.dec2.add_module('FinalConv', nn.Conv2d(data_channels, data_channels, 
        #                                                         kernel_size = 5, stride = 1, padding = 2, bias = False))
        
        
        self.m_out.add_module('Conv1', nn.Conv2d(self.n_latent*self.data_channels, self.data_channels,
                                                 kernel_size = 1, stride = 1, padding = 0, bias = False))
        
        self.m1_out = nn.Sequential()
        self.m2_out = nn.Sequential()
        self.m3_out = nn.Sequential()
        
        self.m1_out.add_module('Conv1', nn.Conv2d(self.n_latent, 1,
                                                 kernel_size = 7, stride = 1, padding = 3, bias = False))
        self.m2_out.add_module('Conv1', nn.Conv2d(self.n_latent, 1,
                                                 kernel_size = 7, stride = 1, padding = 3, bias = False))
        self.m3_out.add_module('Conv1', nn.Conv2d(self.n_latent, 1,
                                                 kernel_size = 7, stride = 1, padding = 3, bias = False))
        
        
        #        'FinalDecode', last_decoding(data_channels*n_latent, data_channels, kernel_size = 5, stride = 1, padding = 2, 
        #                             output_padding = 0, bias = False, drop_rate = 0))
                             
        #self.m_out.add_module('Conv2', nn.Conv2d(self.data_channels, self.data_channels,
        #                                         kernel_size = 3, stride = 1, padding = 1, bias = False))
        
        
        
        #self.lv_out.add_module('Conv1', nn.Conv2d(self.n_latent*self.data_channels, self.data_channels,
        #                                         kernel_size = 7, stride = 1, padding = 3, bias = False))
        
        #self.lv_out.add_module('Conv2', nn.Conv2d(self.data_channels, self.data_channels,
        #                                         kernel_size = 3, stride = 1, padding = 1, bias = False))
        
        self.lv_out.add_module('FullConn1', nn.Linear(2, flatten_dim2 // 2))
        self.lv_out.add_module('Act1', nn.ReLU())
        self.lv_out.add_module('FullConn2', nn.Linear(flatten_dim2 // 2, flatten_dim2))
        self.lv_out.add_module('Reshape1', Reshape((-1, n_features2 , 8, 8)))
        
        n_features = n_features2
        for i, n_layers in enumerate([n_l+2, n_l]):
            block = _DenseBlock(n_layers = n_layers, in_features = n_features, growth_rate = self.K)
            
            self.lv_out.add_module('DecoderDenseBlock%d' % (i+1), block)
            n_features += n_layers*self.K
            
            dec = _Transition(in_features = n_features, out_features = n_features // 2,
                              down = False)
            self.lv_out.add_module('DecodeUp%d' % (i+1), dec)
            n_features = n_features // 2
            
                            
        final_decode = last_decoding(n_features, data_channels, kernel_size = 4, stride = 2, padding = 1, 
                                     output_padding = 1, bias = False, drop_rate = 0)
                
        self.lv_out.add_module('FinalDecode', final_decode)
        self.lv_out.add_module('FinalConv', nn.Conv2d(data_channels, data_channels, 
                                                                 kernel_size = 5, stride = 1, padding = 2, bias = False))
        

    def encoder(self, x):
        zmu = torch.cat((self.cast_m1(self.enc1(x)), self.cast_m2(self.enc2(self.enc1(x)))), 1)
        zlogvar = torch.cat((self.cast_lv1(self.enc1(x)), self.cast_lv2(self.enc2(self.enc1(x)))), 1)
        return zmu, zlogvar
    
    def decoder(self, z):
        #xmu = self.m_out(torch.cat((self.dec1(torch.unsqueeze(z[:,0],-1)), self.dec2(torch.unsqueeze(z[:,1],-1))), 1))
        int_out = torch.cat((self.dec1(torch.unsqueeze(z[:,0],-1)), self.dec2(torch.unsqueeze(z[:,1],-1))), 1)
        m1_in = torch.cat((torch.unsqueeze(int_out[:,0,:,:],1), torch.unsqueeze(int_out[:,3,:,:],1)),1)
        m2_in = torch.cat((torch.unsqueeze(int_out[:,1,:,:],1), torch.unsqueeze(int_out[:,4,:,:],1)),1)
        m3_in = torch.cat((torch.unsqueeze(int_out[:,2,:,:],1), torch.unsqueeze(int_out[:,5,:,:],1)),1)
        xmu = torch.cat((self.m1_out(m1_in), self.m2_out(m2_in), self.m3_out(m3_in)), 1)
        xlogvar = self.lv_out(z)#torch.cat((self.dec1(torch.unsqueeze(z[:,0],-1)), self.dec2(torch.unsqueeze(z[:,1],-1))), 1))
        #xlogvar = self.dec_logvar
        return xmu, xlogvar

    def forward(self, x):
        zmu, zlogvar = self.encoder(x)
        z = self._reparameterize(zmu, zlogvar)
        
        xmu, xlogvar = self.decoder(z)
        
        return zmu, zlogvar, z, xmu, xlogvar
    
    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).type_as(mu)
        return mu + std * eps
    
    def gaussian_log_prob(self, x, mu, logvar):
        return -0.5*(math.log(2*math.pi) + logvar + (x-mu)**2/torch.exp(logvar))
        
    def compute_kld(self, zmu, zlogvar):
        ##0.5*(zmu**2 + torch.exp(zlogvar) - 1 - zlogvar)#
        return 0.5*(zmu**2/torch.exp(self.prior_logvar) + torch.exp(zlogvar)/torch.exp(self.prior_logvar) - 1 - zlogvar + self.prior_logvar)#0.5*(2*math.log(0.25)- 0.5*torch.sum(zlogvar, 1) - 2 + 1/0.25*torch.sum(zlogvar.mul(0.5).exp_(), 1) + torch.sum((0.5-zmu)**2, 1))#
    
    def compute_loss(self, x):
        #freebits = 0
        zmu, zlogvar, z, xmu, xlogvar = self.forward(x)
        l_rec = -torch.sum(self.gaussian_log_prob(x, xmu, xlogvar), 1)
        l_reg = self.compute_kld(zmu, zlogvar)#torch.sum(F.relu(self.compute_kld(zmu, zlogvar) - freebits*math.log(2)) + freebits * math.log(2), 1)#
        return zmu, zlogvar, z, xmu, xlogvar, l_rec, l_reg
    
    def update_beta(self, beta, rec, nu, tau):
        def H(d):
            if d > 0:
                return 1.0
            else:
                return 0.0

        def f(b, d, t):
            return (1-H(d))*math.tanh(t*(b-1)) - H(d)

        return beta*math.exp(nu*f(beta, rec, tau)*rec)
    
    def compute_dis_score(self, p, z):
        # compute disentanglement score where p are true parameter samples and z are latent samples
        if p.is_cuda:
            p = p.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
        else:
            p = p.detach().numpy()
            z = z.detach().numpy()
        
        score = 0
        for i in range(z.shape[1]):
            m = np.concatenate((z[:,i].reshape((-1,1)), p), axis = 1)
            m = np.transpose(m)
            c = np.cov(m)
            score = np.max(np.abs(c[0,1:]))/np.sum(np.abs(c[0,1:])) + score
        
        return score / z.shape[1]
    
        
        
        
        
        
        
        
        
        
        
        



