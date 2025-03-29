# Reference-less decomposition of highly multimode fibers using a physics-driven neural network

# Software requirements

This code is supported for Linux. The code has been tested on the following system:

Linux: Debian 12
Windows: Win 11


# Hardware Requirements

The code requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 4 GB of RAM. For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB
CPU: 4+ cores, 3.3+ GHz/core

It is recommend to use a GPU with > = 4 GB of RAM to accelerate the process, for example, NVIDIA GTX 1060 or later.


# Python Dependencies:

    pip==23.2.1
    mat73==0.62 \
    matplotlib==3.7.2 \
    numpy==1.25.1 \
    scipy==1.11.1 \
    torch==2.0.1

# Package Installation

It is recommended to use a virtual environment for testing.
1. Install Anaconda
2. Create a virtual environment with Python version = 3.9.
3. Install all necessary packages
4. Run the demo

# Demo

In this demo, we print the optimization process and the final correlation between inputs and predictions.


# Use of data

All training results can be saved in a .mat file and can be uploaded to MATLAB for further analysis and visualization.
