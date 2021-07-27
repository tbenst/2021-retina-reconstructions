# 2021-retina-reconstructions
Code for the paper, "Reconstruction of visual images from murine retinal ganglion cell spiking activity using convolutional neural networks"

## Acquiring the data
In order to clone the data with this repository, please have [git-lfs](https://git-lfs.github.com/) installed.

The data contains each image and response for all 2800 images presented to the retina (FEI Faces dataset) in a format convenient for machine learning. The retina response is spike sorted data, binned at 1ms for convenient manipulation. Please get in touch if you're interested in acquiring data with more precise spike timing, or the raw voltage traces (very large file)

## Required software
### Python
All models were trained in Python. A recent version of Python is preferred (e.g. Python 3.8). Required packages include PyTorch, PyTables, Matplotlib, Numpy, and OpenCv2.

### Julia
Most figures were compiled using Julia's Compose.jl. To run final-figures.jl, you will need IterTools, Cairo, Colors, Compose, Fontconfig, PyCall, StatsBase, Glob, FileIO, Measures, Format, Unitful, and Images. Julia 1.6 is recommended.

## training models
The directory `final_outputs` contains jupyter notebooks for training each of the models using PyTorch / Nvidia CUDA acceleration. Depending on your environment, you may wish to comment out the line containing `CUDA_VISIBLE_DEVICES`, or otherwise select a different GPU. You may also want to update `hdf5 = tables.open_file('captures/fei/R1_E3_AMES_200min_200f_14l_rgb.h5','r')` to reflect the path on your local machine, which is relative to the jupyter root; eg `data/R1_E3_AMES_200min_200f_14l_rgb.h5`.

## Questions? Help needed?
Please feel free to file an issue and we are happy to help with getting the code running. If there is demand / interest, we can also help with getting a CoLab notebook up and running.
