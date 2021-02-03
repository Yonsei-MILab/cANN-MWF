# import libraries

import os
from scipy import io

import matplotlib.pyplot as plt
import numpy as np

from utils4cmpxtest import *

# device selection
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2" 

# Load in-vivo test data
meas = io.loadmat('./DATA.mat')

# TE of in-vivo data [s]
te = meas['te']
te = np.squeeze(te[:len(te)])

# mGRE data & create mask
im = meas['aimc']
im = im[:,:,:,:len(te)]
im_mask = create_mask(im)

[ay,ax,az,ae] = im.shape

# model define
model = call_cmpx_model(te)

# trained weight upload
model.load_state_dict(torch.load("./model.pth"))
model.eval()

MWF,uncert_map = test_invivo_cmpx(te,im,model)
MWF = MWF*im_mask
uncert_map = uncert_map*im_mask

io.savemat('./Result.mat',{'MWF':MWF,'uncert_map':uncert_map})
