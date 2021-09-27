"""
@author: Christian Jacobsen, University of Michigan

VAE configuration file

"""

import numpy as np
import torch



def lr_schedule_0(epoch):
    # used for VAE w/out HP network
    e0 = 5900
    if epoch < e0:
        return 0.00075
    else:
        return 0.0001
    
def lr_schedule_1(epoch):
    # used for VAE w/ HP network (more sensitive to learning rate)
    e0 = 6499
    e1 = 12500
    if epoch <= e0:
        return 0.0003
    elif epoch <= e1:
        return 0.00005
    else:
        return 0.000025


# dataset and save paths ----------------------------------------------------------------------------------------------
n_latent = 2                # latent dimension
arch_num = 1               # architecture number
        
train_data_dir = 'data/DarcyFlow/multimodal/kle2_lhs512_bimodal_2.hdf5'   # training data directory
test_data_dir = 'data/DarcyFlow/multimodal/kle2_mc512_bimodal_2.hdf5'     # testing data directory

save_dir = './DarcyFlow/p2/multimodal/arch{}/n{}'.format(arch_num, n_latent) # specify a folder where all similar models belong. 
                                     #    after training, model and configuration will be saved in a subdirectory as a .pth file
continue_training = False           # specify if training is continued from a saved model
continue_path = './n2_kle2_VAEs/DenseVAE/HierarchicalPrior/bimodal_gen_2/DenseVAE_n2_kle2_hp_32.pth'                # the path to the previously saved model                
save_interval = None
# architecture parameters ---------------------------------------------------------------------------------------------

HP = False                 # include heirarchical prior network?

dense_blocks = [4, 6, 4]    # vector containing dense blocks and their length
growth_rate = 4             # see dense architecture for detailed explantation. dense block growth rate
data_channels = 3           # number of input channels
initial_features = 2        # see dense architecture for explanation. Features after initial convolution


if HP:
    prior = 'N/A'           # no need for prior specification w/ HP
    full_param = False      # specifies if the prior network variances are constant (False) or parameterized by NNs (True)
else:
    prior = 'scaled_gaussian'      # specify the prior:
                            #   'std_norm' = standard normal prior (isotropic gaussian)
                            #   'scaled_gaussian' = Factorized Gaussian prior centered at origin.
omega = 0.#40*np.pi/180           # rotation angle of latent space
# training parameters --------------------------------------------------------------------------------------------------

wd = 0.                     # weight decay (Adam optimizer)
batch_size = 64             # batch size (training)
test_batch_size = 512       # not used during training, but saved for post-processing
beta0 = 0.000000001         # \beta during reconstruction-only phase

nu = 0.005
tau = 1                     # these are parameters for the beta scheduler, more details in paper

if HP:                      # specify the learning rate schedule
    lr_schedule = lr_schedule_1
    epochs = 200#14000
    rec_epochs = 100#6500
    if full_param:
        beta_p0 = beta0
        
else:
    lr_schedule = lr_schedule_0
    epochs = 4000 # 6500
    rec_epochs = 2000 # 4000







