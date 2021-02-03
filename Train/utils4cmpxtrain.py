# import libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets

import matplotlib.pyplot as plt
import numpy as np
import math

########################### network model ##########################

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self,input_r, input_i):
        return self.fc_r(input_r)-self.fc_i(input_i), self.fc_r(input_i)+self.fc_i(input_r)

def complex_relu(input_r,input_i):
    return F.relu(input_r), F.relu(input_i)

def complex_dropout3(input_r,input_i):
    dropout = nn.Dropout(p = 0.3)
    output_r = dropout(input_r)
    output_i = input_i
    output_i[output_r==0]=0
    return output_r, output_i

def call_cmpx_model(te):

    use_cuda = torch.cuda.is_available()

    class MWF_Model(nn.Module):

        def __init__(self,input_dim,hidden_dim):
            super(MWF_Model, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.fc_layers = nn.ModuleList()
            self.fc_layers.append(ComplexLinear(len(te), 250))
            for i in range(hidden_dim):
                self.fc_layers.append(ComplexLinear(250, 250))
            self.fc_layers.append(ComplexLinear(250, 1))
            self.fc_layers.append(nn.Linear(2, 30))
            self.fc_layers.append(nn.Linear(30, 1))
            
            # gpu setting
            if use_cuda:
                for layer in self.fc_layers[:-1]:
                    layer = layer.cuda()

        def forward(self,x):

            xr = x[:,:len(te)]
            xi = x[:,len(te):]

            for layer in self.fc_layers[:hidden_dim+1]:
                xr,xi = layer(xr,xi)
                xr,xi = complex_relu(xr,xi)
                xr,xi = complex_dropout3(xr,xi)

            xr,xi = self.fc_layers[hidden_dim+1](xr,xi)
            x = torch.cat((xr, xi), 1)
            x = F.relu(self.fc_layers[hidden_dim+2](x))
            x = self.fc_layers[hidden_dim+3](x)

            return x

    hidden_dim = 10    
    model = MWF_Model(len(te),hidden_dim).cuda().float()
    
    return model

########################### line regression ###########################

def qr_householder(A):
    m, n = A.shape
    Q = np.eye(m) 
    R = A.copy() 

    for j in range(n):
        x = R[j:, j]
        normx = np.linalg.norm(x)
        rho = -np.sign(x[0])
        u1 = x[0] - rho * normx
        u = x / u1
        u[0] = 1
        beta = -rho * u1 / normx

        R[j:, :] = R[j:, :] - beta * np.outer(u, u).dot(R[j:, :])
        Q[:, j:] = Q[:, j:] - beta * Q[:, j:].dot(np.outer(u, u))
        
    return Q, R

def line_regression(data):
    m, n = data.shape
    A = np.array([data[:,0], np.ones(m)]).T
    b = data[:, 1] 

    Q, R = qr_householder(A) 
    b_hat = Q.T.dot(b) 

    R_upper = R[:n, :]
    b_upper = b_hat[:n]  
    
    x = np.linalg.solve(R_upper, b_upper) 
    slope, intercept = x 
    
    return slope, intercept

########################### Signal generation  #########################

class T1_cmpx_Datagen(data.Dataset):
    def __init__(self,TR,FA,te,snr):
        self.TR = TR
        self.FA = FA
        self.te = te
        self.snr = snr
    
    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        
        TE = self.te
        
        ######################## T1 compen setting ##########################
        
        tr_ms = self.TR
        fa_angle = self.FA
        
        TR = tr_ms * 0.001 # TR  
        FA = fa_angle * math.pi*1/180  # FA 

        coscos = math.cos(FA)
        sinsin = math.sin(FA)

        T1my = 350*0.001 #350ms
        T1in = T1ex = 1100*0.001 #1100ms
        
        FT11 = sinsin * (1-np.exp(-TR / T1my)) / (1-coscos * np.exp(-TR / T1my))
        FT12 = sinsin * (1-np.exp(-TR / T1in)) / (1-coscos * np.exp(-TR / T1in))
        FT13 = sinsin * (1-np.exp(-TR / T1ex)) / (1-coscos * np.exp(-TR / T1ex))
        
        ######################## simulation data ############################
        
        tmp_x=np.ones((len(TE)*2,),dtype=np.complex)
        tmp_y=np.ones((1,))
        
        MWF = np.random.rand(1)*0.5
        M0 = MWF
        scaling = 0.5+1.5*np.random.rand(1)
        M2 = (1-MWF)/(1+scaling)
        M1 = M2*scaling
        
        T20, T21, T22 = (np.multiply(0.001,np.random.normal(10, 1)),np.multiply(0.001,np.random.normal(72,10)),np.multiply(0.001,np.random.normal(48, 6)))  
        R20, R21, R22 = (1/T20,1/T21,1/T22)
        
        fex = np.random.normal(0, 50)
        fme = np.random.randint(-20,20)
        fae = np.random.randint(-10,10)
        
        fmy = fex+fme
        fax = fex+fae

        cmpx_my = np.complex(R20,2*np.pi*fmy)
        cmpx_ax = np.complex(R21,2*np.pi*fax)
        cmpx_ex = np.complex(R22,2*np.pi*fex)
        
        signal = M0*FT11*np.exp(-TE*cmpx_my) +M1*FT12*np.exp(-TE*cmpx_ax) +M2*FT13*np.exp(-TE*cmpx_ex)
        
        ############### add noise ########################

        snr = self.snr
        sstd = signal[0]/snr
        noise_sig = signal+sstd*np.random.randn(len(TE))+sstd*np.random.randn(len(TE))*1j

        ########################### normalize ######################
        
        # 1. magnitude normalization
        norm_sig = noise_sig/np.abs(noise_sig[0])
        mag_sig = np.abs(norm_sig)
        
        # 2. phase normalization
        phase = np.unwrap(np.angle(noise_sig))
        
        # calculate linear phase
        phase_set = np.concatenate((np.expand_dims(TE,axis=1),np.expand_dims(phase,axis=1)),axis=1)
        slope, intercept = line_regression(phase_set)
        linear_phase = slope*TE+intercept
        
        phase_sig = phase - linear_phase + np.pi/4
        
        # combine mag,phase to real,imag
        real_sig = mag_sig*np.cos(phase_sig)
        imag_sig = mag_sig*np.sin(phase_sig)

        tmp_x = np.concatenate((real_sig,imag_sig),axis=0)
        tmp_y = MWF
        
        return tmp_x, tmp_y
    
########################### model train ##############################

def train(model, optimizer, criterion, args, training_generator, validation_generator):

    use_cuda = torch.cuda.is_available()
    
    for epoch in range(args.epoch):
        train_loss = 0.0
        val_loss = 0.0
        iteration = 0
        
        # Training
        for data, target in training_generator:

            model.train()

            if use_cuda:
                data = data.cuda().float()
                target = target.cuda().float()

            # 1. clear the gradients of all optimized variables
            optimizer.zero_grad()

            # 2. forward pass
            output = model(data)

            # 3. calculate the loss
            loss = criterion(output, target)

            # 4. backward pass
            loss.backward()

            # 5. parameter update
            optimizer.step()

            # update training loss
            train_loss += loss.item()
            iteration += 1
        
        # Validation
        for val_data, val_target in validation_generator:

            model.eval()

            if use_cuda:
                val_data = val_data.cuda().float()
                val_target = val_target.cuda().float()

            output = model(data)

            loss_val = criterion(output, target)

            # update training loss
            val_loss += loss_val.item()

        # calculate average loss over one epoch
        train_loss = train_loss/iteration
        val_loss = val_loss/iteration
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch+1, train_loss, val_loss))
        
    return model
