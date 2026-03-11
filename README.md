# RAKI-Implementation
Python PyTorch implementation of the Deep Learning MRI Reconstruction Algorithm RAKI
The orginal paper outlining RAKI algorithm: Akçakaya, M., Moeller, S., Weingärtner, S. and Uğurbil, K. (2018). Scan-specific robust artificial-neural-networks for k-space interpolation (RAKI) reconstruction: Database-free deep learning for fast imaging. Magnetic Resonance in Medicine, 81(1), pp.439–453. doi:https://doi.org/10.1002/mrm.27420

# Algorithm details
Main script - Parallel_Imaging initalises and trains as well as evaluates the RAKI algorithm using the Shepp-logan Phantom and child brain MRI png image. The algorithm is robust to different size ACS regions and different acceleration factors (R)

# Changes to the original implementation 
The original description of RAKI in the paper differs to the version of RAKI released by the authors. I have kept to the version as closesly resembling their released implementation as possible. The released version of RAKI can be found here, the biggest difference between my implementation and the version released by the authors is that my implementation uses PyTorch and theres uses TensorFlow. As well as this our algorithms are structued differently

Parameters and important things to note: learning rate = 3*10**-3, momentum = default, number of CNNs = 2*num_coils, activation function = leaky ReLU 
