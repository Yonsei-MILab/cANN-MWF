# Import libraries

import os
from scipy import io
import argparse
from utils4cmpxtrain import *

# device selection
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"  

# in-vivo data
meas = io.loadmat('DATA.mat')

# TE of in-vivo data [s]
te = meas['te']
te = np.squeeze(te[:30])

# repetition time (TR) and flip angle (FA) of in-vivo data [ms,degree]
TR = 46
FA = 25

# model define
model = call_cmpx_model(te)

# Parameters setting
parse = argparse.ArgumentParser()
args = parse.parse_args("")

args.optim = 'Adam'
args.lr = 0.0001
args.epoch = 1000
args.batch_size = 2000

if args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    raise ValueError('Optimiser Error')
    
# loss function
criterion = nn.MSELoss()

# Signal generation Parameters
params = {'shuffle': True, 'batch_size' : args.batch_size}

# Generators
training_set = T1_cmpx_Datagen(TR,FA,te,snr=150)
training_generator = data.DataLoader(training_set,**params)

validation_set = T1_cmpx_Datagen(TR,FA,te,snr=150)
validation_generator = data.DataLoader(validation_set, **params)

# Model Train
model = train(model, optimizer, criterion, args, training_generator, validation_generator)

# save model weights
savePath = "./model.pth"
torch.save(model.state_dict(), savePath)
