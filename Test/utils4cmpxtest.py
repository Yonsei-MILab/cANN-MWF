# import libraries

import torch 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt
from scipy import signal as sg
import numpy as np


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

########################### Testing  #########################

def uncertainity_estimate(x, model, iters, lengthscale):
    model.train()
    
    x_output = np.hstack([model(x).cpu().detach().numpy() for i in range(iters)])
    y_mean = np.mean(x_output,axis=1)
    x2_output = x_output**2
    y_sqrt_mean = np.mean(x2_output,axis=1)
    
    tau = 0.1 * lengthscale**2 / (2. * x.shape[1] * 1e-4)
    y_variance = y_sqrt_mean - y_mean**2
    y_std = np.sqrt(y_variance)
    
    return y_mean, y_std

def test_invivo_cmpx(te,im,model):

    [ay,ax,az,ae] = im.shape
    
    MWF = np.zeros([ay,ax,az])
    uncert_map = np.zeros([ay,ax,az])

    for sslice in range(az):

        im1 = np.squeeze(im[:,:,sslice:sslice+1,:len(te)])

        im2 = np.zeros([ay,ax,ae],dtype=np.complex)
        
        for i in range(ay):
            for j in range(ax):
                sig = im1[i,j,:]/np.abs(im1[i,j,0])
                mag_sig = np.abs(sig)
                phase = np.unwrap(np.angle(sig))

                phase_set = np.concatenate((np.expand_dims(te,axis=1),np.expand_dims(phase,axis=1)),axis=1)
                slope, intercept = line_regression(phase_set)
                linear_phase = slope*te+intercept
                
                phase_sig = phase - linear_phase + np.pi/4
                real_sig = mag_sig*np.cos(phase_sig)
                imag_sig = mag_sig*np.sin(phase_sig)
                im2[i,j,:] = real_sig + 1j*imag_sig

        im3 = np.concatenate((np.real(im2),np.imag(im2)),axis=2)
        im3 = np.reshape(im3,[ay*ax,ae*2])
        im3 = torch.from_numpy(im3)
        im3 = im3.type(torch.FloatTensor)
        im3 = im3.cuda()

        iters_uncertainty = 200
        lengthscale = 1

        y_mean, y_std = uncertainity_estimate(im3, model, iters_uncertainty, lengthscale)

        y_mean2 = np.reshape(y_mean,[ay,ax])
        y_std2 = np.reshape(y_std,[ay,ax])

        MWF[:,:,sslice] = y_mean2
        uncert_map[:,:,sslice] = y_std2

    return MWF,uncert_map

######################### plot & create mask ###############################

def create_mask(im):
    
    im = np.abs(im)

    [ay,ax,az,ae] = im.shape
    im_mask = np.ones([ay,ax,az])
    
    for zz in range(az):
        aim = np.squeeze(im[:,:,zz,:]);
        [ay,ax,ae] = aim.shape  
        mask=np.abs(aim[:,:,0]); mask[mask<np.mean(mask[:])*.5]=0; mask[mask>0]=1;
        mask = np.float32(mask)
        mask=sg.medfilt2d(sg.medfilt2d(mask,kernel_size=17),kernel_size=17)
        im_mask[:,:,zz] = mask
  
    return im_mask


def MWF_plot(MWF,sslice):
    
    fig = plt.figure(figsize=(7,7))
    plt.imshow(np.squeeze(MWF[:,:,sslice]), cmap='hot',vmin=0,vmax=0.2)
    plt.axis('off')

    plt.show()
    
def uncert_plot(uncert_map,sslice):
    
    fig = plt.figure(figsize=(7,7))
    plt.imshow(np.squeeze(uncert_map[:,:,sslice]), cmap='jet',vmin=0,vmax=0.07)
    plt.axis('off')

    plt.show()
