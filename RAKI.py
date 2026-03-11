## RAKI Class for RAKI implementation

# import all nessersary modules/libraries
import imageio.v2 as iio
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom  # import test image
import numpy as np
from scipy import fftpack
from scipy import signal
import Recon_functions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

# arguments

class RAKI(nn.Module):
    # constructor for the CNN
    def __init__(self, kspace, ACS_data, R, epochs=1000):
        # initalising parent class constructor
        super(RAKI, self).__init__()

        # convert to numpy arrays for later computation
        self.kspace_real = torch.from_numpy(kspace.real)
        self.kspace_imag = torch.from_numpy(kspace.imag)
        self.ACS_data_real = torch.from_numpy(ACS_data.real)
        self.ACS_data_imag = torch.from_numpy(ACS_data.imag)

        # find dimensions of kspace and ACS (coils and cols will be the shape for both)
        coils, n_rows, n_cols = np.shape(kspace)
        n_ACS_rows, n_ACS_cols = np.shape(ACS_data[0])

        # equivalent kernel size for GRAPPA
        kernel_size = [4, 5]

        # number of input channels for the Convolutional Neural Network
        num_inputs = 2 * coils

        # number of output channels for the Convolutional Neural Network
        num_outputs = R-1

        # first convolutional layer is a weak GRAPPA weight convolution
        # padding = valid
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=(2, 5), padding="valid", bias=False, dilation=(R, 1)) # dilation R,1 padding 1,2

        # second convolutional layer compresses feature information across coils whilst retaining spatial information
        self.conv2 = nn.Conv2d(32, 8, kernel_size=(1, 1), padding="valid", bias=False, dilation=(R, 1)) # dilation R,1 padding 0

        # third convolutional layer computes the R-1 weights for each coil
        self.conv3 = nn.Conv2d(8, num_outputs, kernel_size=(2, 3), padding="valid", bias=False, dilation=(R, 1)) # dilation R,1 padding 1

        # set up the activiation function
        self.relu = nn.LeakyReLU(0.01)

    # function for initialising the weights of the different CNN networks
    def initialize(self):
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='linear')

    # function: forward propagation through network
    def forward(self, x):
        # input the image into the 1st convolutional layer of the CNN
        x = self.relu(self.conv1(x))
        # input the result of the 1st convolutional layer into the 2nd convolutional layer
        x = self.relu(self.conv2(x))
        # input the result of the 2nd convolutional layer into the 3rd convolutional layer
        x = self.conv3(x)
        # return the output of the 3rd convolutional layer
        return x



                        
            
