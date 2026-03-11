# main script for RAKI reconstruction algorithm

# import all nessersary modules/libraries
import imageio.v2 as iio
import matplotlib
from matplotlib.pyplot import suptitle
from torch.fft import ifftshift
matplotlib.use('QtAgg')
matplotlib.rcParams['image.cmap'] = 'viridis'
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom  # import test image
import numpy as np
from scipy import fftpack
from scipy import signal
import RAKI
import Recon_functions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
from tqdm import trange

# import test image, convert to numpy array and find dimensions
#img = iio.imread("childMRI.png")
img = shepp_logan_phantom()
img_np = np.array(img)
Ny,Nx = np.shape(np.array(img))

# plot the true image
plt.figure()
plt.imshow(img_np)

## Coil Implementation

# coil parameters
n_coils = 8
nx_coils = 2
#coil_x_pos = [100,200]
coil_x_pos = [150,250]
# print(coil_x_pos)
ny_coils = 4
#coil_y_pos = np.linspace(50,200,ny_coils)
coil_y_pos = np.linspace(50,350,ny_coils)
sigma = 50

# creating coils
coil_sensitivities = Recon_functions.creating_coils(Nx, Ny, coil_y_pos, coil_x_pos, sigma)

# plotting sensitivities
Recon_functions.display_images(coil_sensitivities, 'Coil sensitivities')

# multiply the image by the sensitivities
img_np = np.array(img)
coil_view = []
# iterate over the coil sensitivities to multiply the k-space to get the coil views
for i in enumerate(coil_sensitivities):
    # each individual coil view
    x = np.multiply(i[1],img_np)
    coil_view.append(x)
coil_view = np.array(coil_view,dtype=complex)

# display all of the different coil views
Recon_functions.display_images(coil_view, 'Coil view image domain')

# reconstruct coil views to return result of parallel image
plt.figure()
plt.suptitle('Sum-of-squares Coil view reconstruction', fontsize=16)
parallel_image = Recon_functions.sum_of_squares(n_coils,np.fft.fftshift(np.fft.fft2(coil_view)))
plt.imshow(abs(parallel_image))
# Recon_functions.display_images(abs(parallel_image), 'Sum-of-squares Coil view reconstruction')

# k-space transformations

# apply noise and display
coil_view_kspace = Recon_functions.applying_noise(img_np, coil_view)
Recon_functions.display_images(np.log(np.abs(coil_view_kspace)), 'Coil view with noise')

print(len(coil_view_kspace[0])) # print the number of rows in kspace

## Parallel imaging: Sampling K-Space coil views with ACS ##

# acceleration rate (number of phase encoding lines sampled)
R = 2
# undersampling K-Space and getting ACS data
num_ACS_rows = 24
norm_factor = 0.015 / np.max(abs(coil_view_kspace[:]))
coil_view_kspace = np.multiply(coil_view_kspace, norm_factor)
usamp_kspace, full_ACS, ACS_row_min, ACS_row_max, usamp_ACS_data, Ny, Nx = Recon_functions.undersampling(coil_view_kspace, R, Ny, Nx, num_ACS_rows)
# display the undersampled coils
Recon_functions.display_images(np.array([np.fft.ifft2(np.fft.ifftshift(j)) for j in usamp_kspace]), 'undersampled coils')

# RAKI preprocessing

# push the network to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# initalise all objects necessary for RAKI reconstruction
model = RAKI.RAKI(usamp_kspace, full_ACS, R).to(device)
model.initialize()

# array of all ACS rows indices
ACS_rows = np.arange(num_ACS_rows)
# array of all k-space rows
kspace_rows = np.arange(Ny)

# split kspace and ACS_data coils into seperate real and imaginary components
kspace = np.concatenate((np.real(usamp_kspace), np.imag(usamp_kspace)), axis=0)
ACS_data_full = np.concatenate((np.real(full_ACS), np.imag(full_ACS)), axis=0)
kspace = torch.tensor(np.array(kspace), dtype=torch.float).unsqueeze(0)
ACS_data_full = torch.tensor(np.array(ACS_data_full), dtype=torch.float).unsqueeze(0)

# the GRAPPA equivalent kernel size
kernel_size = [5, 4]
# the kernel sizes of each of the layers of the convolutional neural network
[kernel_x_1, kernel_y_1] = [5, 2]
[kernel_x_2, kernel_y_2] = [1, 1]
[kernel_x_3, kernel_y_3] = [3, 2]
# calculate how much the x (column) decreases after all 3 convolutional layers
x_conv_decrease_total = np.int32(kernel_x_1 - 1 + kernel_x_2 - 1 + kernel_x_3 - 1) # for odd kernel sizes, size decreases by size // 2
# floor(k/2) gives left decrease
x_conv_decrease_left = x_conv_decrease_total // 2
# ceil(k/2) gives right decrease
x_conv_decrease_right = np.int32(np.ceil(x_conv_decrease_total / 2))
# calculate the starting index of x (columns) after decrease for slicing
target_x_start = x_conv_decrease_left # account for the python 0 indexing
# calculate the final index of x (columns) after decrease for slicing
target_x_end = np.int32(Nx - x_conv_decrease_right) # Nx is the size to turn it to an index -1 for python 0 indexing
# calculate how much the y (rows) decreases after all 3 convolutional layers
y_conv_decrease_total = np.int32(R * (kernel_y_1 - 1 + kernel_y_2 - 1 + kernel_y_3 - 1)) # for even kernel sizes (with dilation R) size decreases by R * size / 2
# floor(k/2) gives top decrease
y_conv_decrease_top = y_conv_decrease_total // 2
# ceil(k/2) gives right decrease
y_conv_decrease_bottom = np.int32(np.ceil(y_conv_decrease_total / 2))
# calculate the starting index of y (rows) after decrease for slicing
target_y_start = y_conv_decrease_top # account for the python 0 indexing
# calculate the final index of y (rows) after decrease for slicing
target_y_end = np.int32(num_ACS_rows - y_conv_decrease_bottom) # num_ACS_rows is the size to turn it to an index -1 for python 0 indexing

# calculate the dimension sizes after convolutional decrease
target_dim_X = np.int32(Nx - x_conv_decrease_total)
target_dim_Y = np.int32(num_ACS_rows - y_conv_decrease_total)
target_dim_Z = R - 1

# create targets needed for evaluation
targets = torch.zeros((2*n_coils, target_dim_Z, target_dim_Y, target_dim_X))
for coil in range(2*n_coils):
    for offset in range(0, R - 1):
        targets[coil, offset, :, :] = ACS_data_full[:, coil, target_y_start+offset+1 : target_y_end+offset+1, target_x_start : target_x_end]                    # +1 for inclusive end

# sanity check: verify target dimensions match network output
with torch.no_grad():
    test_out = model(ACS_data_full).squeeze(0)   # shape: [R-1, actual_Y, actual_X]
actual_Y, actual_X = test_out.shape[-2], test_out.shape[-1]

# if mismatch, re-derive targets using actual output shape (on-the-fly)
if actual_Y != target_dim_Y or actual_X != target_dim_X:
    print(f"Warning: Recomputing targets using actual network output shape "
          f"Y={actual_Y} (expected {target_dim_Y}), X={actual_X} (expected {target_dim_X})")
    target_dim_Y, target_dim_X = actual_Y, actual_X
    targets = torch.zeros((2*n_coils, R-1, target_dim_Y, target_dim_X))
    for coil in range(2*n_coils):
        for offset in range(0, R - 1):
            targets[coil, offset, :, :] = ACS_data_full[:, coil,target_y_start+offset+1 : target_y_start+offset+1+target_dim_Y,target_x_start : target_x_start+target_dim_X]

###### RAKI Implementation #######

# normalise inputs and outputs (target and test)
ACS_data_full = ACS_data_full.to(device)
kspace = kspace.to(device)
interpolated_kspace = kspace.squeeze(0)
interpolated_kspace = interpolated_kspace.to(device)
targets = targets.to(device)

# number of individual forward and back propagations during training of CNNs
epochs = 300

loss_per_network = []
num_offset_rows = int(Ny / R)
num_ACS_offset_rows = int(num_ACS_rows / R)

# iterate across 2*nc coil views (real+imaginary)
for i in trange(2*n_coils, desc="Processing"):
    # instantiate the RAKI neural network class
    model = RAKI.RAKI(usamp_kspace, full_ACS, R).to(device)
    # initalise the weights for the model
    model.initialize()
    model.train()
    # model weights optimizer definition
    optimizer = optim.Adam(model.parameters(), lr=3*10**-3)
    loss_per_epoch = [] # initalise the loss array to hold all of the networks training losses

    # training the RAKI network
    # iterate training over a total number of epochs
    for epoch in range(epochs):
        # forward pass through the network
        optimizer.zero_grad()
        # training occurs on entirety of ACS
        output = model(ACS_data_full)
        output = output.squeeze(0)

        # loss = loss_fun(output, targets[i,:,:,:])
        loss = Recon_functions.loss_fun(targets[i,:,:,:], output, device)
        loss_per_epoch.append(loss.detach().item())
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()

    loss_per_network.append(loss_per_epoch)
    # perform predictions on the full k-space
    model.eval()
    with torch.no_grad():
        # perform on k-space
        output_kspace = model(kspace).to(device).squeeze(0)
        for offset in range(0, R - 1):
            output_slice = output_kspace[offset, ::R, :]  # shape: [n_rows_out, n_cols_out]
            n_rows_out, n_cols_out = output_slice.shape

            # x: anchor at target_x_start, width taken directly from output
            t_x_start = target_x_start
            t_x_end = t_x_start + n_cols_out

            # y: anchor at first missing row for this offset, span taken from output row count
            t_y_start = target_y_start + offset + 1
            t_y_end = t_y_start + n_rows_out * R  # R spacing between sampled rows

            interpolated_kspace[i, t_y_start: t_y_end: R, t_x_start: t_x_end] = output_slice


# reconstructing the undersampled k-space
complex_kspace = Recon_functions.RAKI_complex_recombination(interpolated_kspace, n_coils)
complex_kspace = complex_kspace / norm_factor

# create a numpy array of all of the final images for reconstruction
image_batch = np.array([np.abs(np.fft.ifft2(np.fft.ifftshift(complex_kspace[i]))) for i in range(n_coils)])
image_batch2 = np.array([np.log(np.abs(complex_kspace[i])+10**-6) for i in range(n_coils)])

Recon_functions.display_images(image_batch, f"Image space Reconstruction (R={R})")
Recon_functions.display_images(image_batch2, f"k-space Reconstruction (R={R})")

final_recon_image = Recon_functions.sum_of_squares(n_coils, complex_kspace)

# print(output.shape)

# plotting the loss for the neural networks
plt.figure()
for i in range(2*n_coils):
    plt.subplot(4,4,i+1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(epochs), loss_per_network[i])

plt.figure()
plt.title(f"RAKI reconstruction (R={R})")
plt.imshow(final_recon_image)

plt.figure()
plt.subplot(1,2,1)
plt.title('RAKI reconstruction', fontsize=16)
plt.imshow(final_recon_image)
plt.subplot(1,2,2)
plt.title('Ground truth', fontsize=16)
plt.imshow(parallel_image)

plt.show()
