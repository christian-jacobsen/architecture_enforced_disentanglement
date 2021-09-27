"""
@author: Christian Jacobsen, University of Michigan

VAE training file: requires a configuration file "config_train.py" to train

"""

from config_train import *
from VAE_arch1_train import *
from VAE_arch3_train import *
import os
import time
import numpy as np



if __name__ == '__main__':
    
    start = time.time()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
        
    
    file_exists = True
    trial_num = 0
    if False:
        if HP:
            while file_exists:
                if trial_num > 100:
                    break
                save_dir_temp = save_dir + '/VAE_HP_' + str(trial_num)
                filename = 'VAE_HP_' + str(trial_num) + '.pth'
                path_exists = os.path.exists(save_dir_temp)
                if path_exists:
                    file_exists = os.path.exists(save_dir_temp + '/' + filename)
                else:
                    file_exists = False
                    os.mkdir(save_dir_temp)
                trial_num += 1
        else:
            while file_exists:
                if trial_num > 100:
                    break
                save_dir_temp = save_dir + '/VAE_' + str(trial_num)
                filename = 'VAE_' + str(trial_num) + '.pth'
                path_exists = os.path.exists(save_dir_temp)
                if path_exists:
                    file_exists = os.path.exists(save_dir_temp + '/' + filename)
                else:
                    file_exists = False
                    os.mkdir(save_dir_temp)
                trial_num += 1
    else:
        while file_exists:
            if trial_num > 100:
                break
            save_dir_temp = save_dir + '/VAE_' + str(trial_num)
            filename = 'VAE_' + str(trial_num) + '.pth'
            path_exists = os.path.exists(save_dir_temp)
            if path_exists:
                file_exists = os.path.exists(save_dir_temp + '/' + filename)
            else:
                file_exists = False
                os.mkdir(save_dir_temp)
            trial_num += 1
    
    save_dir = save_dir_temp
    
    if False:
        if HP:
            DenseVAE_HP_train(train_data_dir, test_data_dir, save_dir, filename, \
                              epochs, rec_epochs, batch_size, test_batch_size, wd, beta0, lr_schedule, nu, tau, \
                              full_param, data_channels, initial_features, dense_blocks, growth_rate, n_latent, \
                              continue_training, continue_path, omega, i, save_interval)
            
        else:
            DenseVAE_train(train_data_dir, test_data_dir, save_dir, filename, \
                           epochs, rec_epochs, batch_size, test_batch_size, wd, beta0, lr_schedule, nu, tau, \
                           data_channels, initial_features, dense_blocks, growth_rate, n_latent, \
                           prior)
    else:
        if arch_num == 1:
            VAE_arch1_train(train_data_dir, test_data_dir, save_dir, filename, \
                                 epochs, rec_epochs, batch_size, test_batch_size, wd, beta0, lr_schedule, nu, tau, \
                                 data_channels, initial_features, growth_rate, n_latent, \
                                 prior)
        elif arch_num == 2:
            print('arch 2 does not exist yet')
        elif arch_num == 3:
            VAE_arch3_train(train_data_dir, test_data_dir, save_dir, filename, \
                                 epochs, rec_epochs, batch_size, test_batch_size, wd, beta0, lr_schedule, nu, tau, \
                                 data_channels, initial_features, growth_rate, n_latent, \
                                 prior)
        
    end = time.time()
    
    print('------------ Training Completed --------------')
    print('Elapsed Time: ', (end-start)/60, ' mins')
    print('Save Location: ', save_dir)



