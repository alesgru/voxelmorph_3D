"""
VoxelMorph testing
"""
# import from notebooks
from IPhyton_import import NotebookFinder
import sys
sys.meta_path.append(NotebookFinder())


# python imports
import os
import glob
import random
import numpy as np
from argparse import ArgumentParser

#local imports
from model import cvpr2018_net, SpatialTransformer
import datagenerators
import losses

#external imports
import torch
import nibabel as nib
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path



def test(gpu,
         size,
         data_dir,
         atlas_dir, 
         model, 
         init_model_file,
         saveDir,
         nr_val_data):
    """
    model testing function
    :param gpu: integer specifying the gpu to use
    :param size: integer related to desired size of the input images
    :param data_dir: String describing the location of the data
    :param atlas_dir: String where atlases are located
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from
    :param saveDir: String specifiying the direction to store the outputs
    :param nr_val_data: the number of validation samples must corresponed to the valiable for the train function
    """

    #set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
   
    #set size
    vol_size = np.array([size,size,size])

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    
    #sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    sim_loss_fn = losses.mse_loss

    # Set up model
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)
    model.load_state_dict(torch.load(init_model_file, map_location=lambda storage, loc: storage))

    # Use this to warp segments
    trf = SpatialTransformer(vol_size, mode='nearest')
    trf.to(device)
    test_strings = glob.glob(os.path.join(data_dir, '*.nii'))
    test_strings = test_strings[:nr_val_data]
    test_strings = [i for i in test_strings if "L1-L3" in i]
    #test_strings = ['.\\Data\\Train\\64\\3YQ_unhealthyL2_L1-L3.nii','.\\Data\\Train\\64\\x1b_unhealthyL2_L1-L3.nii' ]
    mean_val_loss = 0
    #iteration over the test volumes
    for k in range(0, len(test_strings)):
        
        #load the data create fixed and moving image
        X_vol, atlas_vol = datagenerators.load_example_by_name(test_strings[k], atlas_dir, size)
        
        input_fixed  = torch.from_numpy(atlas_vol).to(device).float()
        input_fixed  = input_fixed.permute(0, 4, 1, 2, 3)
        
        input_moving  = torch.from_numpy(X_vol).to(device).float()
        input_moving  = input_moving.permute(0, 4, 1, 2, 3)
        
        #produce the warp field
        warp, flow = model(input_moving, input_fixed)
    
        flow_vectors = flow[0]
        shape = flow_vectors.shape
        
        #plot the middle slice of the vector field
        #flow_vectors  = flow_vectors.permute(1, 2, 3, 0)
        #flow_vectors = flow_vectors.detach().cpu().numpy()
        #flow_vectors_middle = flow_vectors[:,:,int(shape[2]/2),0:2]
        #print('shape')
        #print(flow_vectors_middle.shape)
        #flow_vectors_middle = flow_vectors_middle.squeeze()
        #fig, axes = neuron.plot.flow([flow_vectors_middle], width=5,show = False)
        #print(type(fig))
        #fig.savefig(os.path.join(get_outputs_path(), test_strings[k][len(data_dir):-4] +'.png'))

        
        # generate the new sample with the vector field
        warped   = trf(input_moving, flow).detach().cpu().numpy()
        
        mean_val_loss += sim_loss_fn(warp, input_fixed)
        
        warped = np.squeeze(warped)
        #print(warped.shape)
        #plot_middle_slices(warped)
        
        
        # store the generated volume
        img = nib.Nifti1Image(warped, np.eye(4))
        nib.save(img,os.path.join(saveDir, test_strings[k][len(data_dir):]))
    mean_val_loss /= len(test_strings)
    print("Mean validation loss: ")
    print(mean_val_loss)

        #vals, labels = dice(warp_seg, atlas_seg, labels=good_labels, nargout=2)
        #dice_vals[:, k] = vals
        #print(np.mean(dice_vals[:, k]))
        #print(np.mean(vals))

        #return


#test(0, 128, 'data/PMSD_voxelmorph/128/','/data/PMSD_voxelmorph/atlas128/', 'vm2', '/outputs/agrund/PMSD_voxelmorph/1000.ckpt')

#-----------------------------------------------Parser------------------------------------------------------

import warnings
from argparse import ArgumentParser
if __name__ == "__main__":
    experiment = Experiment()
    print('EXPERIMENT: ')
    print(experiment)
    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default = '0')
    
    parser.add_argument("--size",
                        type=int,
                        default = '128')
    
    parser.add_argument("--data_dir",
                        type=str,
                        dest="data_dir",
                        default='/data/PMSD_voxelmorph/128/')
    
    parser.add_argument("--atlas_dir",
                        type=str,
                        dest="atlas_dir",
                        default='/data/PMSD_voxelmorph/atlas128/')

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm2',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--init_model_file", 
                        type=str,
                        dest="init_model_file",
                        default = '/outputs/agrund/PMSD_voxelmorph/1000.ckpt',
                        help="model weight file")
    
    parser.add_argument("--saveDir", 
                        type=str,
                        dest="saveDir",
                        default = get_outputs_path())
   
    parser.add_argument("--nr_val_data", 
                        type=int,
                        dest="nr_val_data",
                        default = 4)


    args, unknown = parser.parse_known_args()
    
    test(**vars(args))







