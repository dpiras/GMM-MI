import sys
from os.path import isfile
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits

sys.path.append('../models')
import models

eps = 1e-5
wave_resolution = 0.01 #A
desiredMinW = 3785 #A
desiredMaxW = 6910 #A
L = 327680
padL = 7589
padR = 7590
left_last_zero = 7588
right_first_zero = 320090
mid_first_zero = 159453
mid_last_zero = 162893


def infer(args):

    #--- Device settings
    device = torch.device('cuda:%d'%args.gpu if args.gpu >= 0 else 'cpu')

    #--- Setup the network
    model = models.ae1d().to(device)
    network_data = torch.load(args.pretrained,map_location=device)
    model.load_state_dict(network_data['state_dict'],strict=True)

    #--- Load the spec and prepare it
    hdu = fits.open(args.fits_file)
    wave = hdu[1].data.field('WAVE').astype(np.float32).T
    flux = hdu[1].data.field('FLUX').astype(np.float32).T

    #- Trim
    flux = flux[ np.logical_and(wave>=(desiredMinW-eps),wave<=(desiredMaxW+eps)) ]
    wave = wave[ np.logical_and(wave>=(desiredMinW-eps),wave<=(desiredMaxW+eps)) ]

    #- Pad
    flux = np.pad(flux,pad_width=(padL,padR),mode='constant',constant_values=(0,0))
    wave = np.pad(wave,pad_width=(padL,padR),mode='constant',constant_values=(0,0))

    #- MedNorm
    mask = np.ones((1,L),dtype=np.float32)
    mask[0,:left_last_zero+1] = 0
    mask[0,right_first_zero:] = 0
    mask[0,mid_first_zero:mid_last_zero+1] = 0
    flux /= np.median(flux*mask)
    
    #--- Forward pass
    model.eval()
    data = torch.from_numpy(flux[None,None,...]).to(device)

    with torch.no_grad():
        output = model.forward(data).detach().cpu().numpy()
        data_ = data.detach().cpu().numpy()

    #--- Visualize
    plt.figure()
    ax = plt.gca()

    st = wave_resolution
    WAVE = np.arange(desiredMinW-padL*st,desiredMaxW+.001+padR*st,step=st)
    ax.plot(WAVE,data_.squeeze(),'b',label='input')
    ax.plot(WAVE,output.squeeze(),'r',label='reconst')
    ax.legend()
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained', type=str, default = '../models/model_128d_e182_i1500000.pth.tar')
    parser.add_argument('--fits_file', type=str, default = 'sample.fits')
    parser.add_argument('--gpu', metavar='gpu', default=0, type=int)
    args = parser.parse_args()
   
    infer(args)
