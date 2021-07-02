# Python code for ModalPINN

## Context

Python code to define, train and output ModalPINNs as defined in the paper

> _ModalPINN : an extension of Physics-Informed Neural Networks with enforced truncated Fourier decomposition for periodic flow reconstruction using a limited number of imperfect sensors._ Gaétan Raynaud, Sébastien Houde, Frédérick P. Gosselin (2021) 

## How to run basic jobs

This code is designed to be launched on a computationel cluster (initially for Compute Canada - Graham server) using the following batch commands:

    #!/bin/bash
    #SBATCH --gres=gpu:t4:1
    #SBATCH --nodelist=gra1337
    #SBATCH --cpus-per-task=2
    #SBATCH --mem=50G
    #SBATCH --job-name=ModalPINN
    #SBATCH --time=0-10:00
    
    module load python/3.7.4
    source ~/ENV/bin/activate
    python ./ModalPINN_VortexShedding.py --Tmax 9 --Nmes 5000 --Nint 50000 --multigrid --Ngrid 5 --NgridTurn 200 --WidthLayer 25 --Nmodes 3 
    deactivate

For each job launched, a folder is created in ./OutputPythonScript and is identified by date-time information. In this folder, the content of consol prints is saved in a out.txt file alongside other files (mode shapes, various plots...) including the model itself in a pickle archive.

Please refer to the help for the arguments sent to the parser and to the next section for librairies requirements.

## Requirements

This code has been tested and used with the following versions of environments and libraires and may not be directly compatible with other versions (especially with tensorflow>=2.0)

Environments in Compute Canada

    StdEnv/2020
    nixpkgs/16.09
    python/3.7.4
     
Python libraries

    numpy==1.17.4
    scipy==1.3.2
    tensorflow_gpu==1.14.1
    matplotlib==3.1.1
     
For a more detailed list of python libraries, see requirements.txt. You can also set up an environment with

    virtualenv --no-download ~/ENV
    source ~/ENV/bin/activate
    pip install --upgrade pip
    python pip install -r requirements.txt

## Licence

Codes are provided under licence GNU GPL v3.
