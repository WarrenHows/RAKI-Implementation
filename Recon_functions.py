# Python file containing all functions relating to reconstruction for parallel imaging

# import all relvant modules
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import multiprocessing as mp
import RAKI

# function: sum of squares reconstruction
def sum_of_squares(n_coils, coil_kspace):
    # initalising image for iteration
    image_total = np.zeros_like(coil_kspace[0], dtype=float)
    # for loop to reconstruct the full GRAPPA image (sum of squares)
    for i in range(n_coils):
        # squeeze into 1 dimension
        k_recon = np.squeeze(coil_kspace[i, :, :])
        # iteratively improve the image with the square of the new coil information
        image_total += np.abs((np.fft.ifft2(np.fft.ifftshift(k_recon)))) ** 2

    # sqrt the final iteration to retrieve the final image
    final_image = image_total ** 0.5
    return final_image

# function: display image input
def display_images(image_batch, title=''):
    # initialise new figure
    plt.figure()
    # assign title to the new figure
    plt.suptitle(title, fontsize=16)
    # find length of the batch
    length = image_batch.shape[0]
    # if there is only 1 image
    if length == 1:
        plt.imshow(image_batch)
        plt.colorbar()
    elif length == 2:
        plt.subplot(1, 2, 1)
        plt.imshow(image_batch[0])
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(image_batch[1])
        plt.colorbar()
    elif length == 3:
        plt.subplot(1, 3, 1)
        plt.imshow(image_batch[0])
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(image_batch[1])
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.imshow(image_batch[2])
        plt.colorbar()
    else:
        # initalise the number of rows for batch
        rw = 1
        # if length is longer than 3
        if length > 3:
            rw = length // 2 + 1 # calculate the row
        # iterate over the number of images in the batch
        for i in range(length):
            plt.subplot(rw, (length // 2), i + 1)
            plt.imshow(abs(image_batch[i]))
            plt.colorbar()

# function: Undersample inputted k-space
def undersampling(coil_view_kspace, R, Ny, Nx, num_ACS = 24):
    # initalise sampling of image
    full_ACS = []
    usamp_kspace = []
    usamp_ACS_data = []

    # find centre row based on odd or even size
    if len(coil_view_kspace[0]) % 2 == 0:
        center_row = int(len(coil_view_kspace[0]) / 2) - 1
    else:
        center_row = int(len(coil_view_kspace[0]) // 2)

    # ACS range
    abv = int(num_ACS / 2)
    bel = int(num_ACS / 2) - 1
    ACS_row_min = center_row - abv
    ACS_row_max = center_row + bel

    # shape changes neccersarry for RAKI
    coils,Ny,Nx = coil_view_kspace.shape
    if Ny % R != 0:
        Ny_new = (Ny // R) * R
        Nx_new = (Nx // R) * R
        coil_view_kspace = coil_view_kspace[:Ny_new,:Nx_new]
    else:
        Ny_new = Ny
        Nx_new = Nx

    # sampling K-Space for each coil (with ACS)
    for i in coil_view_kspace:
        # full K-Space sampling
        y = np.zeros_like(i, dtype=complex)
        y[::R] = i[::R] # sample at the acceleration rate
        usamp_kspace.append(y)
        # including ACS lines with K-Space
        temp = i[ACS_row_min:ACS_row_max + 1,:]
        usamp_ACS = np.zeros_like(temp, dtype=complex)
        usamp_ACS[::R] = temp[::R]
        usamp_ACS_data.append(usamp_ACS)
        full_ACS.append(i[ACS_row_min:ACS_row_max + 1, :])

    usamp_kspace = np.array(usamp_kspace, dtype=complex)
    full_ACS = np.array(full_ACS)
    usamp_ACS_data = np.array(usamp_ACS_data)

    full_ACS = full_ACS[:,:,:Nx_new]

    return usamp_kspace, full_ACS, ACS_row_min, ACS_row_max, usamp_ACS_data, Ny_new, Nx_new

# create a mathematical equivalent to parallel reciever coils
def creating_coils(Nx, Ny, coil_y_pos, coil_x_pos, sigma, noise=0):
    # Create spatial grid
    x = np.linspace(0, Nx, Nx)
    y = np.linspace(0, Ny, Ny)
    X, Y = np.meshgrid(x, y)

    # assigning the coordiantes of each coils position
    coil_pos = []
    for i in coil_y_pos:
        for j in coil_x_pos:
            coil_pos.append((i, j))
    # print("coil_pos ",coil_pos)
    coil_pos = np.array(coil_pos)

    # assigning coil sensitivity with Gaussian distribution originating from coil position
    coil_sensitivities = []
    for y0, x0 in coil_pos:
        # calculating the Gaussian distibution at certain coil locations with (imaginary) phase distribution
        G_r = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * (sigma ** 2)))
        G1im = 1j * np.exp(-((X - x0 - 5) ** 2 + (Y - y0 + 10) ** 2) / (2 * (sigma ** 2)))
        coil_sensitivities.append(G_r + G1im)
    coil_sensitivities = np.array(coil_sensitivities, dtype=complex)

    return coil_sensitivities

# function: applying noise to the ground truth
def applying_noise(img_np, coil_view):
    # perform fourier transform for entire image
    img_kspace = np.fft.fftshift(np.fft.fft2(img_np))
    # initialise empty array
    coil_view_kspace = []
    # iterate over all of the coil views
    for index, i in enumerate(coil_view):
        # fourier transform the individual views of the coil
        x = np.fft.fftshift(np.fft.fft2(i))
        # define the stnadard deviation of the noise
        sigma_noise = 0  # 0-15: Low, 15-30: Medium, 30>: High
        # define the noise
        noise = np.random.normal(0, sigma_noise, x.shape)
        # define the imaginary noise component
        noise_im = np.random.normal(0, sigma_noise, x.shape) * 1j
        # adding noise to K-Space
        noise = noise + noise_im
        x += noise
        # append to list of coil view of images containing noise
        coil_view_kspace.append(x)
    # turn list into a numpy array
    coil_view_kspace = np.array(coil_view_kspace, dtype=complex)

    return coil_view_kspace

# function: combines the real and imaginary components into complex k-space data
def RAKI_complex_recombination(interpolated_kspace, n_coils):
    # isolate real and imaginary components
    real_kspace = interpolated_kspace[:n_coils,:,:]
    imag_kspace = interpolated_kspace[n_coils:,:,:]
    # comnine the real and imaginary components
    complex_kspace = real_kspace + (imag_kspace * 1j)
    # convert to numpy array
    complex_kspace = complex_kspace.detach().numpy()
    # return numpy array of the complex interpolated k-space from RAKI
    return complex_kspace

# function: loss function is the Frobenius norm
def loss_fun(targets, ACS_recon, device):
    diff = ACS_recon - targets
    loss = torch.norm(diff, p="fro").to(device)
    return loss