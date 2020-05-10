#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
*Preliminary* pytorch implementation.
VoxelMorph testing
"""
# python imports
from IPhyton_import import NotebookFinder
import sys
sys.meta_path.append(NotebookFinder())
   
import os
import glob
import random
import warnings
import sys
from argparse import ArgumentParser
from torchvision.transforms import ToTensor, Resize, Compose


# external imports
import numpy as np
import nibabel as nib
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

# internal imports
from model import cvpr2018_net
import datagenerators
import losses
    
def train(gpu,
          data_dir,
          size,
          atlas_dir,
          lr,
          n_iter,
          data_loss,
          model,
          reg_param, 
          batch_size,
          n_save_iter,
          model_dir,
          nr_val_data):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param size: int desired size of the volumes: [size,size,size]
    :param atlas_dir: direction to atlas folder
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    :param nr_val_data: number of validation examples that should be separated from the training data
    """
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    vol_size = np.array([size,size,size])
    # Get all the names of the training data
    vol_names = glob.glob(os.path.join(data_dir, '*.nii'))
    #random.shuffle(vol_names)
    #test_vol_names =  vol_names[-nr_val_data:]
    test_vol_names =  vol_names[:nr_val_data]
    #test_vol_names = [i for i in test_vol_names if "L2-L4" in i]
    print('these volumes are separated from the data and serve as validation data : ')
    print(test_vol_names)
    
    
    #train_vol_names = vol_names[:-nr_val_data]
    train_vol_names = vol_names[nr_val_data:]
    #train_vol_names = [i for i in train_vol_names if "L2-L4" in i]

    random.shuffle(train_vol_names)
    writer = SummaryWriter(get_outputs_path())

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:
        raise ValueError("Not yet implemented!")

    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, atlas_dir, size, batch_size)
    

    # Training loop.
    for i in range(n_iter):

        # Save model checkpoint and plot validation score
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)
            # load validation data
            val_example_gen = datagenerators.example_gen(test_vol_names, atlas_dir, size, 4)
            val_data = next(val_example_gen)
            val_fixed  = torch.from_numpy(val_data[1]).to(device).float()
            val_fixed  = val_fixed.permute(0, 4, 1, 2, 3)
            val_moving = torch.from_numpy(val_data[0]).to(device).float()
            val_moving = val_moving.permute(0, 4, 1, 2, 3)
            
            #create validation data for the model
            val_warp, val_flow = model(val_moving, val_fixed)
            
            #calculte validation score
            val_recon_loss = sim_loss_fn(val_warp, val_fixed) 
            val_grad_loss = grad_loss_fn(val_flow)
            val_loss = val_recon_loss + reg_param * val_grad_loss
            
            #tensorboard
            writer.add_scalar('Loss/Test', val_loss , i)
            
            #prints
            print('validation')
            print("%d,%f,%f,%f" % (i, val_loss.item(), val_recon_loss.item(), val_grad_loss.item()), flush=True)
    

        # Generate the moving images and convert them to tensors.
        
        data_for_network = next(train_example_gen)
        input_fixed  = torch.from_numpy(data_for_network[1]).to(device).float()
        input_fixed  = input_fixed.permute(0, 4, 1, 2, 3)
        input_moving = torch.from_numpy(data_for_network[0]).to(device).float()
        input_moving = input_moving.permute(0, 4, 1, 2, 3)

        # Run the data through the model to produce warp and flow field
        warp, flow = model(input_moving, input_fixed)
        print("warp_and_flow_field")
        print(warp.size())
        print(flow.size())

        # Calculate loss
        recon_loss = sim_loss_fn(warp, input_fixed) 
        grad_loss = grad_loss_fn(flow)
        loss = recon_loss + reg_param * grad_loss
        
        #tensorboard
        writer.add_scalar('Loss/Train', loss , i)
        print("%d,%f,%f,%f" % (i, loss.item(), recon_loss.item(), grad_loss.item()), flush=True)

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

#-------------------------------  Parser   -------------------------------    
import warnings
from argparse import ArgumentParser
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    #create polyaxon experiment
    experiment = Experiment()
    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")

    parser.add_argument("--data_dir",
                        type=str,
                        default = "/data/PMSD_voxelmorph/128/")
                        #help="data folder with training vols")

    parser.add_argument("--atlas_dir",
                        type=str,
                        dest="atlas_dir",
                        default='/data/PMSD_voxelmorph/atlas128/')
                        #help="gpu id number")
    parser.add_argument("--size",
                        type=int,
                        dest="size",
                        default=128,
                        help="size of the images")
    
    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-4,
                        help="learning rate")

    parser.add_argument("--n_iter",
                        type=int,
                        dest="n_iter",
                        default=10000,
                        help="number of iterations")

    parser.add_argument("--data_loss",
                        type=str,
                        dest="data_loss",
                        default='mse',
                        help="data_loss: mse of ncc")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm2',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--lambda", 
                        type=float,
                        dest="reg_param", 
                        default=0.01,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")

    parser.add_argument("--batch_size", 
                        type=int,
                        dest="batch_size", 
                        default=1,
                        help="batch_size")

    parser.add_argument("--n_save_iter", 
                        type=int,
                        dest="n_save_iter", 
                        default=300,
                        help="frequency of model saves")

    parser.add_argument("--model_dir", 
                        type=str,
                        dest="model_dir", 
                        default=get_outputs_path(),
                        help="models folder")
    
    parser.add_argument("--nr_val_data", 
                        type=int,
                        dest="nr_val_data",
                        default = 4)


    args, unknown = parser.parse_known_args()
   
   
    train(**vars(args))

