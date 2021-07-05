# Python code for ModalPINN

## Context

Python code to define, train and output ModalPINNs as defined in the paper

> _ModalPINN : an extension of Physics-Informed Neural Networks with enforced truncated Fourier decomposition for periodic flow reconstruction using a limited number of imperfect sensors._ Gaétan Raynaud, Sébastien Houde, Frédérick P. Gosselin (2021) 

Contact email : gaetan.raynaud (at) polymtl.ca 

## Files

The main files are :
- *ModalPINN_VortexShedding.py* : performs flow reconstruction using ModalPINN with dense or sparse data, possibly with out of synchronisation or gaussian noise (see the argument to the parser). It requires:
    - *Load_train_data_desync.py*:
        Python file containing functions that extract and prepare data for training and validation.
    - *NN_functions.py*:
        Python file containing functions specific to neural networks, optimisers and plots.
- *ClassicPINN_VortexShedding.py* : define a PINN for performance comparison with ModalPINN on dense data. It requires:
    - *NN_functions_classicPINN.py* which translates the same functions than in NN_functions.py but without the modal approach
- *reactions_process.py* and *text_flow.py* provided by M. Boudina to read data sets (see the training data section below)

## How to run basic jobs

This code is designed to be launched on a computationel cluster (initially for Graham server on Compute Canada) using the following batch commands:

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

For each job launched, a folder is created in ./OutputPythonScript and is identified by date-time information. In this folder, the content of console prints is saved in *out.txt* alongside other files (mode shapes, various plots...) including the model itself in a pickle archive.

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
     
For a more detailed list of python libraries, see *requirements.txt*. You can also set up an environment with

    virtualenv --no-download ~/ENV
    source ~/ENV/bin/activate
    pip install --upgrade pip
    python pip install -r requirements.txt

## Training data

Training data presented in the paper was provided by Boudina et al. 
> Boudina, M., Gosselin, F., & Étienne, S. (2021). Vortex-induced vibrations: A soft coral feeding strategy? Journal of Fluid Mechanics, 916, A50. doi:10.1017/jfm.2021.252 

and is available for download on Zenodo
> Boudina, Mouad. (2021). Numerical simulation data of a two-dimensional flow around a fixed circular cylinder [Data set]. Zenodo. http://doi.org/10.5281/zenodo.5039610  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5039610.svg)](https://doi.org/10.5281/zenodo.5039610)

along with two python scripts (*reactions_process.py* and *text_flow.py*) that perform the reading of these data files.

Nonetheless other data can be used. Provided functions in *Load_train_data_desync.py* might be reused if the structure of data suits [time step,element id] for u,v,p and a list of x,y [element id]. Otherwise, it might be necessary to adapt these functions to your data structure.

## Provided results and import of previously trained ModalPINN

Some of the trained ModalPINN which results are plotted in the main paper are saved in the folders OutputPythonScript. Wieghts and biases values of the model are stored in a pickle archive and can be imported by using these lines  

    repertoire= 'OutputPythonScript/Name_of_the_folder'
    filename_restore = repertoire + '/DNN2_40_40_2_tanh.pickle' # Attention to change the name of .pickle depending of the NN layers
    w_u,b_u,w_v,b_v,w_p,b_p = nnf.restore_NN(layers,filename_restore)

instead of these:

    w_u,b_u = nnf.initialize_NN(layers)
    w_v,b_v = nnf.initialize_NN(layers)
    w_p,b_p = nnf.initialize_NN(layers)

## Licence

Codes are provided under licence GNU GPL v3.
