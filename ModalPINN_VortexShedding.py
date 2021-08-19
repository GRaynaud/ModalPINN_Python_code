# -*- coding: utf-8 -*-
"""
ModalPINN Python Code
This is the main Python file for performing flow reconstruction using ModalPINN
as described in the paper

    ModalPINN : an extension of Physics-Informed Neural Networks with enforced 
    truncated Fourier decomposition  for periodic flow reconstruction using a 
    limited number of imperfect sensors. 
    G. Raynaud, S. Houde, F. P. Gosselin (2021)

This file contains the main losses functions of the ModalPINN as well as the 
main steps of the training. Nonetheless, it calls functions from 
-Load_train_data_desync.py:
    Python file containing functions that extract and prepare data for training 
    and validation.
-NN_functions.py:
    Python file containing functions specific to
        o neural networks (construction, initialisation),
        o optimisers (calling from scipy or tf interfaces, initialisation, training steps),
        o plots.

This file is designed to be launched on a computationel cluster (initially for 
Compute Canada - Graham server) using the following batch commands:
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

For each job launched, a folder is created in ./OutputPythonScript and is 
identified by date-time information. In this folder, the content of consol prints 
is saved in a out.txt file alongside other files (mode shapes, various plots...)
including the model itself in a pickle archive.

Please refer to the help for the arguments sent to the parser and to the readme 
for librairies requirements.


@author: Gaétan Raynaud. 
ORCID : orcid.org/0000-0002-2802-7366
email : gaetan.raynaud (at) polymtl.ca
"""

# =============================================================================
# Librairies Import
# =============================================================================

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime
import os
import pickle
from shutil import copyfile
import sys
import GPUtil
import time
import argparse 
from tensorflow.python.client import device_lib

# Code parts
import NN_functions as nnf
import Load_train_data_desync as ltd

# Link to simulations data 
# In the paper, we used those from Boudina et al. (2020) that can be downloaded 
# at https://zenodo.org/record/5039610
filename_data = 'Data/fixed_cylinder_atRe100'

t0 = time.time()
# =============================================================================
# matplotlib parameters
# =============================================================================

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=18)
plt.rc('figure',titlesize=24)


# =============================================================================
# Preparing the writing of console prints in out.txt
# =============================================================================

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

# =============================================================================
# File copy and folder creation
# Here we create a folder containing all the data of this job
# And we copy current python files to keep track of how the job was launched
# =============================================================================
r = np.int(np.ceil(1000*np.random.rand(1)[0])) # This random number is used in case 2 jobs are launched at the exact same time so that the newly created folders does not merge the one into the other
d = datetime.datetime.now()
pythonfile = os.path.basename(__file__)
repertoire = 'OutputPythonScript/ModalPINN_'+ d.strftime("%Y_%m_%d-%H_%M_%S") + '__' +str(r)
os.makedirs(repertoire, exist_ok=True)
copyfile(pythonfile,repertoire+'/Copy_python_script.py')
copyfile('NN_functions.py',repertoire+'/NN_functions.py')
copyfile('Load_train_data_desync.py',repertoire+'/Load_train_data_desync.py')

f = open(repertoire+'/out.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)
print('File copy and stdout ok')


# Print devices available


list_devices = device_lib.list_local_devices()
print('Devices available')
print(list_devices)

# =============================================================================
# Set arguments passed through bash
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument('--Tmax',type=float,default=None,help="Define the max time allowed for optimisation (in hours)")
parser.add_argument('--Nmodes',type=int,default=2,help="Number of modes, including zero frequency")
parser.add_argument('--Nmes',type=int,default=5000,help="Number of measurement points to provide for optimisation")
parser.add_argument('--Nint',type=int,default=50000,help="Number of computing points to provide for equation evaluation during optimisation")
parser.add_argument('--LossModes',action="store_true",default=False,help="Use of modal equations during optimisation")
parser.add_argument('--multigrid',action="store_true",default=False,help="Use of multi grid")
parser.add_argument('--Ngrid',type=int,default=1,help="Number of batch for Adam optimization")
parser.add_argument('--NgridTurn',type=int,default=1000,help="Number of iterations between each batch changement")
parser.add_argument('--Noise',type=float,default=0.,help="Define standard deviation of gaussian noise added to measurements")
parser.add_argument('--WidthLayer',type=int,default=20,help="Number of neurons per layer and per mode")
parser.add_argument('--SparseData',action="store_true",default=False,help="if activated, use simulated  measurements data for training. Else use dense data")
parser.add_argument('--DesyncSparseData',action="store_true",default=False,help="if activated (and --SparseData == True), then simulated measurements are randomly made out of synchronisation")
parser.add_argument('--TwoZonesSampling',action="store_true",default=False,help="if activated, the sampling of equation penalisation points is carried out using 2 zones (with more points near the cylinder). Else use a uniform sampling")


args = parser.parse_args()

print('Args passed to python script')
print('Tmax '+str(args.Tmax)+' (h)')
print('Nmodes %d' % (args.Nmodes))
print('Nmes %d' % (args.Nmes))
print('Nint %d' % (args.Nint))
print('Use Loss Modes : ' + str(args.LossModes))
print('Multigrid : '+str(args.multigrid))
print('Ngrid : '+str(args.Ngrid))
print('Ngrid Turn : '+str(args.NgridTurn))
print('STD Noise : %.2e' % (args.Noise))
print('Neurons per layer and per mode : %d' % (args.WidthLayer))
print('Sparse Data : ' + str(args.SparseData))
print('Desync Sparse Data : ' + str(args.DesyncSparseData))

if args.TwoZonesSampling:
    IntSampling = '2zones'
else:
    IntSampling = 'uniform'

print('Sampling of V_in : '+IntSampling)

# =============================================================================
# Physical and geometrical parameters 
# =============================================================================

Re = 100.
Lxmin = -4. 
Lxmax = 8. 
Lx = Lxmax-Lxmin
Lymin = -4. 
Lymax = 4. 
Ly = Lymax-Lymin
x_c = 0. # x-Position of the centre of the cylindre
y_c = 0. # y-Position of the centre of the cylindre
r_c = 0.5 # radius
d = 2.*r_c
u_in = 1. 
rho_0 = 1.

omega_0 = 1.036 #Dimensionless frequency

geom = [Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c]


def xbc5(s):
    '''
    Compute cylinders border x coordinate as a function of curvilinear abscissa s \in [0,1]
    input : s (tf tensor, usually of shape [Nbc,1])
    return a tf tensor of the same shape as s
    '''
    return x_c + r_c*tf.cos(2*np.pi*s)
def ybc5(s):
    '''
    Compute cylinders border y coordinate as a function of curvilinear abscissa s \in [0,1]
    input : s (tf tensor, usually of shape [Nbc,1])
    return a tf tensor of the same shape as s
    '''
    return y_c + r_c*tf.sin(2*np.pi*s)

# =============================================================================
# Choix de discretisation
# =============================================================================

Nmodes = args.Nmodes

Nmes = args.Nmes # Number of measurement points in the domain in case of dense data
Nint = args.Nint # Number of points to penalize NS equations in \Omega_f
Nbc = 1000 # Number of points to sample on cylinders norder


multigrid = args.multigrid # If true, Adam optimiser will change of V_in sampling 
# every NgridTurn iterations between the Ngrid generated
Ngrid = args.Ngrid
NgridTurn = args.NgridTurn 

stdNoise = args.Noise # In case of artificially noised data, it defines the 
# standard deviation inputted in the Gaussian distribution

# List of frequencies associated with each mode shapes
# Note that it could be replaced with an arbitrary list of frequencies
# or even tf.Variables() that could be optimized during training
list_omega = np.asarray([k*omega_0 for k in range(Nmodes)]) 

# Structure of each Neural Network that approximate a mode shape
layers = [2,args.WidthLayer*Nmodes,args.WidthLayer*Nmodes,Nmodes]

# =============================================================================
# Training tracking variables
# =============================================================================

global it
global listeErrTimeSerie
global listeErrValidTimeSerie

it=0
listeErrTimeSerie = []
listeErrValidTimeSerie = []

plot_config = False

if args.Tmax==None:
    Tmax = None  #0.5*3600 #8h
else:
    Tmax = 3600*args.Tmax


# =============================================================================
# Placeholders declaration
# In TF<2, one can define placeholders and build an operation graph based on these.
# Values are provided only at the computation in a dictionary tf_dict when running
# session.run(TF quantity that depends on placeholders,feed_dict=TF dictionary containing placeholders values)
# =============================================================================

Nxpitot = 40 # Number of simulated pitot probe locations in the flow (4 sections of 10 points)
Ncyl = 30 # Number of points around the cylinder to simulated pressure probes
Ntimes = 201 # Number of timesteps in simulations data

# Placeholders for V_in (penalization of equations)
x_tf_int = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
y_tf_int = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
t_tf_int = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])

# Placeholders for general fitting data (especially dense data)
x_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
y_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
t_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
u_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
v_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
p_tf_mes = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])

# Placeholder for simulated pitot probe
x_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
y_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
t_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
u_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
v_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1])
p_tf_mes_pitot = tf.compat.v1.placeholder(dtype=tf.float32,shape=[Ntimes*Nxpitot,1]) #  Not really used since only u and v are used at these locations for training


# Preparing desynchronisation of pitot probe. Especially Used if args.DesyncSparseData == True
Delta_phi_np_pitot = 0.*np.random.uniform(low=0.0,high=2*np.pi/omega_0, size=Nxpitot)

if args.DesyncSparseData:
    Delta_t_tf_pitot = tf.Variable(Delta_t_np_pitot,dtype=tf.float32,shape=[Nxpitot])
else:
    Delta_phi_tf_pitot = tf.constant(Delta_phi_np_pitot,dtype=tf.float32,shape=[Nxpitot])


t_tf_mes_pitot_unflatten = tf.reshape(t_tf_mes_pitot,[Ntimes,Nxpitot])
t_tf_mes_pitot_resync_unflatten = tf.convert_to_tensor([[ t_tf_mes_pitot_unflatten[t,k] - Delta_phi_tf_pitot[k] for k in range(Nxpitot)] for t in range(Ntimes)])
t_tf_mes_pitot_resync = tf.reshape(t_tf_mes_pitot_resync_unflatten,[Ntimes*Nxpitot,1]) 

# Cylindre data for simulated pressure probe
x_tf_mes_cyl = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
y_tf_mes_cyl = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
t_tf_mes_cyl = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
p_tf_mes_cyl = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])


# Border
s_tf = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])
one_s_tf = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,1])

# Frequencies
w_tf = tf.constant(list_omega,dtype=tf.float32,shape=[Nmodes])


# =============================================================================
# Model construction
# =============================================================================

# Initialisation of weights w and biases b for each variable u, v and p
w_u,b_u = nnf.initialize_NN(layers)
w_v,b_v = nnf.initialize_NN(layers)
w_p,b_p = nnf.initialize_NN(layers)

## For the restoration of a previous model, comment the 3 previous lines and uncomment the 3 following
## Make sure that parameters in parser are correct (Nmodes,WidthLayer...)
# repertoire= 'OutputPythonScript/Name_of_the_folder'
# filename_restore = repertoire + '/DNN2_40_40_2_tanh.pickle' # Attention to change the name of .pickle depending of the NN layers
# w_u,b_u,w_v,b_v,w_p,b_p = nnf.restore_NN(layers,filename_restore)
 

def fluid_u(x,y):
    '''
    Compute mode shapes of u
    Input : x,y TF tensors of shape [Nint,1]
    Return TF tensor of shape [1,Nint,Nmodes] with complex values
    '''
    return nnf.out_nn_modes_uv(x,y,w_u,b_u,geom)

def fluid_u_t(x,y,t):
    '''
    Compute u at instant t and position x,y
    Input: x,y,t TF tensors of shape [Nint,1]
    Return TF tensor of shape [Nint,1] with real values
    '''
    return nnf.NN_time_uv(x,y,t,w_u,b_u,geom,omega_0)

def fluid_v(x,y):
    '''
    Compute mode shapes of v
    Input : x,y TF tensors of shape [Nint,1]
    Return TF tensor of shape [1,Nint,Nmodes] with complex values
    '''
    return nnf.out_nn_modes_uv(x,y,w_v,b_v,geom)

def fluid_v_t(x,y,t):
    '''
    Compute v at instant t and position x,y
    Input: x,y,t TF tensors of shape [Nint,1]
    Return TF tensor of shape [Nint,1] with real values
    '''
    return nnf.NN_time_uv(x,y,t,w_v,b_v,geom,omega_0)

def fluid_p(x,y):
    '''
    Compute mode shapes of p
    Input : x,y TF tensors of shape [Nint,1]
    Return TF tensor of shape [1,Nint,Nmodes] with complex values
    '''
    return nnf.out_nn_modes_p(x,y,w_p,b_p)

def fluid_p_t(x,y,t):
    '''
    Compute p at instant t and position x,y
    Input: x,y,t TF tensors of shape [Nint,1]
    Return TF tensor of shape [Nint,1] with real values
    '''
    return nnf.NN_time_p(x,y,t,w_p,b_p,omega_0)

# =============================================================================
# Forces on cylinder
# =============================================================================

def force_cylinder_flatten(t):
    '''
    t : tf.float32 tensor shape [Nt,1]  
    ----
    return
    fx_tf,fy_tf :  tf.float32 tensor of shape [Nt,] containing averaged horizontal force on cylinder at time t
    '''
    Nt = int(t.shape[0])
    Ns = 1000 # Number of points to perform the integration over the border
    s_cyl = tf.constant(np.linspace(0.,1.,Ns), dtype = tf.float32, shape = [Ns,1])*tf.transpose(1+0.*t)
    # s_cyl = tf.random.uniform([Ns,1], minval=0., maxval = 1., dtype = tf.float32)*tf.transpose(1+0.*t)
    # Reshaping Space x Times on a same dimension
    s_cyl_r = tf.reshape(s_cyl,[Nt*Ns,1])
    x_cyl_r = tf.reshape(xbc5(s_cyl_r),[Nt*Ns,1]) 
    y_cyl_r = tf.reshape(ybc5(s_cyl_r),[Nt*Ns,1])
    t_cyl = (1.+0*s_cyl)*tf.transpose(t)
    t_cyl_r = tf.reshape(t_cyl,[Nt*Ns,1])
    
    # Computing fluid values along the border
    u = fluid_u_t(x_cyl_r,y_cyl_r,t_cyl_r)
    v = fluid_v_t(x_cyl_r,y_cyl_r,t_cyl_r)
    p = fluid_p_t(x_cyl_r,y_cyl_r,t_cyl_r)
    
    # Computing differentiated quantities
    u_x = tf.gradients(u, x_cyl_r)[0]
    u_y = tf.gradients(u, y_cyl_r)[0]
    u_xx = tf.gradients(u_x, x_cyl_r)[0]
    u_yy = tf.gradients(u_y, y_cyl_r)[0]
    
    v_x = tf.gradients(v, x_cyl_r)[0]
    v_y = tf.gradients(v, y_cyl_r)[0]
    v_xx = tf.gradients(v_x, x_cyl_r)[0]
    v_yy = tf.gradients(v_y, y_cyl_r)[0]
    
    # Computing normal and tangent vectors
    nx_base = - tf.gradients(y_cyl_r, s_cyl_r)[0]
    ny_base = tf.gradients(x_cyl_r, s_cyl_r)[0]
    normalisation = tf.sqrt(tf.square(nx_base) + tf.square(ny_base))
    nx = nx_base/normalisation
    ny = ny_base/normalisation
    
    # Computing local forces elements
    fx_tf_local = -p*nx + 2.*(1./Re)*u_x*nx + (1./Re)*(u_y+v_x)*ny
    fy_tf_local = -p*ny + 2.*(1./Re)*v_y*ny + (1./Re)*(u_y+v_x)*nx
    
    # Reshape to [Ns,Nt]
    fx_tf_local_r2 = tf.reshape(fx_tf_local,[Ns,Nt])
    fy_tf_local_r2 = tf.reshape(fy_tf_local,[Ns,Nt])
    
    # Integrating along the border for every time step
    fx_tf = -2.*np.pi*r_c*tf.reduce_mean(fx_tf_local_r2,axis=0)
    fy_tf = -2.*np.pi*r_c*tf.reduce_mean(fy_tf_local_r2,axis=0)
    
    return fx_tf,fy_tf


# =============================================================================
# Definition of functions for loss
# =============================================================================

def loss_int_mode(x,y):
    '''
    Parameters
    ----------
    x,y : float 32 tensor [Nint,1]
    
    Returns
    -------
    Return a tf.float32 tensor of shape [Nint,1] computing squared errors on modal equations
    '''
    all_u = fluid_u(x,y)
    all_v = fluid_v(x,y)
    all_p = fluid_p(x,y)
    

    one = tf.transpose(0.*x + 1.)
    
    def customgrad(fgrad,xgrad):
        '''
        Input frgad,xgrad : tf.complex64 tensor of shape [1,Nint,N+1] and [1,Nint] resp.
        Return a tf.complex64 tensor df/dx of shape [1,Nint,N+1]
        (tf.gradients does not seem to work with complex values and with f being of order 3... But it is mainly the same thing here)
        '''
        fgrad_xgrad =  [tf.complex(tf.gradients(tf.real(fgrad[:,:,k]), xgrad, grad_ys = one)[0],tf.gradients(tf.imag(fgrad[:,:,k]), xgrad, grad_ys = one)[0]) for k in range(Nmodes)]
        return tf.transpose(tf.convert_to_tensor(fgrad_xgrad), perm=[2,1,0])
    
    all_u_x = customgrad(all_u,x)
    all_u_y = customgrad(all_u,y)
    
    all_v_x = customgrad(all_v,x)
    all_v_y = customgrad(all_v,y)
    
    all_p_x = customgrad(all_p,x)
    all_p_y = customgrad(all_p,y)
    
    all_u_xx = customgrad(all_u_x,x)
    all_u_yy = customgrad(all_u_y,y)
    
    all_v_xx = customgrad(all_v_x,x)
    all_v_yy = customgrad(all_v_y,y)
    
    
    # x axis momentum equation
    f_u = tf.transpose(tf.convert_to_tensor([tf.complex(0.,k*omega_0)*all_u[:,:,k] for k in range(Nmodes)]), perm=[1,2,0])
    f_u += all_p_x
    f_u += (-1./Re)*(all_u_xx + all_u_yy)
    
    f_u_4a = [tf.reduce_sum(tf.convert_to_tensor([all_u[:,:,l]*all_u_x[:,:,k-l] for l in range(k+1)]), axis = 0) for k in range(Nmodes)]
    f_u += tf.transpose(tf.convert_to_tensor(f_u_4a), perm = [1,2,0])
    
    f_u_4b = [tf.reduce_sum(tf.convert_to_tensor([all_v[:,:,l]*all_u_y[:,:,k-l] for l in range(k+1)]), axis = 0) for k in range(Nmodes)]
    f_u += tf.transpose(tf.convert_to_tensor(f_u_4b), perm = [1,2,0])
    
    f_u_5a = [tf.reduce_sum(tf.convert_to_tensor([all_u[:,:,l]*tf.conj(all_u_x[:,:,l-k]) for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_u_5a[-1] = f_u_5a[-2]*0.
    f_u += tf.transpose(tf.convert_to_tensor(f_u_5a), perm=[1,2,0])
    
    f_u_5b = [tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_u[:,:,l-k])*all_u_x[:,:,l] for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_u_5b[-1] = f_u_5b[-2]*0.
    f_u += tf.transpose(tf.convert_to_tensor(f_u_5b), perm=[1,2,0])

    f_u_5c = [tf.reduce_sum(tf.convert_to_tensor([all_v[:,:,l]*tf.conj(all_u_y[:,:,l-k]) for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_u_5c[-1] = f_u_5c[-2]*0.
    f_u += tf.transpose(tf.convert_to_tensor(f_u_5c), perm=[1,2,0])
    
    f_u_5d = [tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_v[:,:,l-k])*all_u_y[:,:,l] for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_u_5d[-1] = f_u_5d[-2]*0.
    f_u += tf.transpose(tf.convert_to_tensor(f_u_5d), perm=[1,2,0])    
    
    
    f_u = tf.reduce_sum(nnf.square_norm(f_u), axis=2)
    
    # y axis Momentum equation
    f_v = tf.transpose(tf.convert_to_tensor([tf.complex(0.,k*omega_0)*all_v[:,:,k] for k in range(Nmodes)]), perm=[1,2,0])
    f_v += all_p_y
    f_v += (-1./Re)*(all_v_xx + all_v_yy)
    
    f_v_4a = [tf.reduce_sum(tf.convert_to_tensor([all_u[:,:,l]*all_v_x[:,:,k-l] for l in range(k+1)]), axis = 0) for k in range(Nmodes)]
    f_v += tf.transpose(tf.convert_to_tensor(f_v_4a), perm = [1,2,0])
    
    f_v_4b = [tf.reduce_sum(tf.convert_to_tensor([all_v[:,:,l]*all_v_y[:,:,k-l] for l in range(k+1)]), axis = 0) for k in range(Nmodes)]
    f_v += tf.transpose(tf.convert_to_tensor(f_v_4b), perm = [1,2,0])
    
    f_v_5a = [tf.reduce_sum(tf.convert_to_tensor([all_u[:,:,l]*tf.conj(all_v_x[:,:,l-k]) for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_v_5a[-1] = f_v_5a[-2]*0.
    f_v += tf.transpose(tf.convert_to_tensor(f_v_5a), perm=[1,2,0])
    
    f_v_5b = [tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_u[:,:,l-k])*all_v_x[:,:,l] for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_v_5b[-1] = f_v_5b[-2]*0.  #quand k=N, k+1 > N
    f_v += tf.transpose(tf.convert_to_tensor(f_v_5b), perm=[1,2,0])

    f_v_5c = [tf.reduce_sum(tf.convert_to_tensor([all_v[:,:,l]*tf.conj(all_v_y[:,:,l-k]) for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_v_5c[-1] = f_v_5c[-2]*0.
    f_v += tf.transpose(tf.convert_to_tensor(f_v_5c), perm=[1,2,0])
    
    f_v_5d = [tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_v[:,:,l-k])*all_v_y[:,:,l] for l in range(k+1,Nmodes)]),axis=0) for k in range(Nmodes)]
    f_v_5d[-1] = f_v_5d[-2]*0.
    f_v += tf.transpose(tf.convert_to_tensor(f_v_5d), perm=[1,2,0])    
    

    f_v = tf.reduce_sum(nnf.square_norm(f_v), axis=2)
    
    
    # Mass conservation equation
    div_u = all_u_x + all_v_y
    div_u = tf.reduce_sum(nnf.square_norm(div_u), axis=2)
    
    return div_u + f_u + f_v


def loss_int_time(x,y,t):
    '''
    Parameters
    ----------
    x,y,t : tf.float 32 tensor [Nint,1]

    Returns
    -------
    Return [Nint,1] tensor containing squared error on NS equations
    '''
    u = fluid_u_t(x,y,t)
    v = fluid_v_t(x,y,t)
    p = fluid_p_t(x,y,t)
    
    u_t = tf.gradients(u,t)[0]
    v_t = tf.gradients(v,t)[0]
    
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    
    v_x = tf.gradients(v, x)[0]
    v_y = tf.gradients(v, y)[0]
    v_xx = tf.gradients(v_x, x)[0]
    v_yy = tf.gradients(v_y, y)[0]
    
    p_x = tf.gradients(p, x)[0]
    p_y = tf.gradients(p, y)[0]

    f_u = u_t + (u*u_x + v*u_y) + p_x - (1./Re)*(u_xx + u_yy) 
    f_v = v_t + (u*v_x + v*v_y) + p_y - (1./Re)*(v_xx + v_yy)
    div_u = u_x + v_y
    
    return tf.square(f_u)+tf.square(f_v)+tf.square(div_u)


def loss_mes(xmes,ymes,tmes,umes,vmes,pmes):
    '''
    xmes,ymes,tmes,umes,vmes,pmes : [Nmes,1] tf.float32 tensor
    Return [Nmes,1] tf.float32 tensor containing square difference to measurements 
    '''
    u_DNN = fluid_u_t(xmes,ymes,tmes)
    v_DNN = fluid_v_t(xmes,ymes,tmes)
    p_DNN = fluid_p_t(xmes,ymes,tmes)
    
    return tf.square(u_DNN-umes) + tf.square(v_DNN-vmes) + tf.square(p_DNN-pmes)

def loss_mes_uv(xmes,ymes,tmes,umes,vmes):
    '''
    xmes,ymes,tmes,umes,vmes : [Nmes,1] tf.float32 tensor
    Return [Nmes,1] tf.float32 tensor containing square difference to measurements of velocity
    '''
    u_DNN = fluid_u_t(xmes,ymes,tmes)
    v_DNN = fluid_v_t(xmes,ymes,tmes)
    
    return tf.square(u_DNN-umes) + tf.square(v_DNN-vmes)

def loss_mes_p(xmes,ymes,tmes,pmes):
    '''
    xmes,ymes,tmes,pmes : [Nmes,1] tf.float32 tensor
    Return [Nmes,1] tf.float32 tensor containing square difference to measurements of pressure
    '''
    p_DNN = fluid_p_t(xmes,ymes,tmes)
    
    return tf.square(p_DNN-pmes)


def loss_BC(s):
    '''
    Return error on u=v=0 on cylinder border for each mode
    Input s : [Nbc,1] tf.float32 tensor of coordinates \in [0,1]
    Output : [] tf.float32 real positive number
    '''    
    x = xbc5(s)
    y = ybc5(s)
    u_k = fluid_u(x,y)
    v_k = fluid_v(x,y)
    
    err = tf.convert_to_tensor([nnf.square_norm(u_k[0,:,k]) + nnf.square_norm(v_k[0,:,k]) for k in range(Nmodes)])
    
    return tf.reduce_sum(tf.reduce_mean(err,axis=1))


# =============================================================================
# Training loss creation
# =============================================================================

# Wrap error on modal equations
Loss_int_mode_wrap = tf.reduce_mean(loss_int_mode(x_tf_int, y_tf_int))

# Wrap error on physical equations
Loss_int_time_wrap = tf.reduce_mean(loss_int_time(x_tf_int, y_tf_int ,t_tf_int))

# Wrap error on (u,v,p) measurements
Loss_dense_mes = tf.reduce_mean(loss_mes(x_tf_mes,y_tf_mes,t_tf_mes,u_tf_mes,v_tf_mes,p_tf_mes))

# Wrap error on (u,v) measurements at simulated pitot probes locations
Loss_mes_pitot = tf.reduce_mean(loss_mes_uv(x_tf_mes_pitot,y_tf_mes_pitot,t_tf_mes_pitot_resync,u_tf_mes_pitot,v_tf_mes_pitot))
Loss_mes_pitot_desync = tf.reduce_mean(loss_mes_uv(x_tf_mes_pitot,y_tf_mes_pitot,t_tf_mes_pitot,u_tf_mes_pitot,v_tf_mes_pitot))

# Wrap error on pressure measurement around cylindre border
Loss_mes_cyl = tf.reduce_mean(loss_mes_p(x_tf_mes_cyl,y_tf_mes_cyl,t_tf_mes_cyl,p_tf_mes_cyl))

# Simulated experimental losses
Loss_mes_exp =  Loss_mes_pitot + Loss_mes_cyl

if args.SparseData:
    Loss_mes = Loss_mes_exp
else: # Dense measurements are used for training
    Loss_mes = Loss_dense_mes

if args.LossModes:
    Loss = Loss_int_mode_wrap + Loss_mes
else: #Physical equations are used instead of modal equations
    Loss = Loss_int_time_wrap + Loss_mes

# =============================================================================
# Optimizer configuration
# =============================================================================

opt_LBFGS = nnf.declare_LBFGS(Loss)

opt_Adam = nnf.declare_Adam(Loss, lr=1e-5)

sess = nnf.declare_init_session()


# =============================================================================
# GPU use before loading data
# =============================================================================
print('GPU use before loading data')
GPUtil.showUtilization()

# =============================================================================
# Data set preparation
# =============================================================================

if args.SparseData:
    # Let's load data only at locations defined for simulated measurements
    print('Loading Sparse Data')
    
    x_int,y_int,t_int,s_train,xmes_pitot,ymes_pitot,tmes_pitot,umes_pitot,vmes_pitot,pmes_pitot,xmes_cyl,ymes_cyl,tmes_cyl,umes_cyl,vmes_cyl,pmes_cyl,Delta_phi_np_pitot_applied = ltd.training_dict(Nmes,Nint,Nbc,filename_data,geom,Tintmax=1e2,data_selection = 'cylinder_pitot',desync=args.DesyncSparseData, multigrid=multigrid,Ngrid=Ngrid,stdNoise=stdNoise,method_int = IntSampling)
    Ncyl = len(xmes_cyl)
    Npitot = len(xmes_pitot)
    Tmin = 400.
    
    
    if multigrid:
        tf_dict = []
        for k in range(Ngrid):
            tf_dict_temp = {x_tf_int : np.reshape(x_int[k],(Nint,1)),
             y_tf_int : np.reshape(y_int[k],(Nint,1)),
             t_tf_int : np.reshape(t_int[k],(Nint,1)),
             s_tf : np.reshape(s_train,(Nbc,1)),
             x_tf_mes_cyl : np.reshape(xmes_cyl,(Ncyl,1)),
             y_tf_mes_cyl : np.reshape(ymes_cyl,(Ncyl,1)),
             p_tf_mes_cyl : np.reshape(pmes_cyl,(Ncyl,1)),
             t_tf_mes_cyl : np.reshape(tmes_cyl,(Ncyl,1)),
             x_tf_mes_pitot : np.reshape(xmes_pitot,(Npitot,1)),
             y_tf_mes_pitot : np.reshape(ymes_pitot,(Npitot,1)),
             u_tf_mes_pitot : np.reshape(umes_pitot,(Npitot,1)),
             v_tf_mes_pitot : np.reshape(vmes_pitot,(Npitot,1)),
             p_tf_mes_pitot : np.reshape(pmes_pitot,(Npitot,1)),
             t_tf_mes_pitot : np.reshape(tmes_pitot,(Npitot,1)),
             }
            tf_dict.append(tf_dict_temp)
        
    else:      
        tf_dict = {x_tf_int : np.reshape(x_int,(Nint,1)),
             y_tf_int : np.reshape(y_int,(Nint,1)),
             t_tf_int : np.reshape(t_int,(Nint,1)),
             s_tf : np.reshape(s_train,(Nbc,1)),
             x_tf_mes_cyl : np.reshape(xmes_cyl,(Ncyl,1)),
             y_tf_mes_cyl : np.reshape(ymes_cyl,(Ncyl,1)),
             p_tf_mes_cyl : np.reshape(pmes_cyl,(Ncyl,1)),
             t_tf_mes_cyl : np.reshape(tmes_cyl,(Ncyl,1)),
             x_tf_mes_pitot : np.reshape(xmes_pitot,(Npitot,1)),
             y_tf_mes_pitot : np.reshape(ymes_pitot,(Npitot,1)),
             u_tf_mes_pitot : np.reshape(umes_pitot,(Npitot,1)),
             v_tf_mes_pitot : np.reshape(vmes_pitot,(Npitot,1)),
             p_tf_mes_pitot : np.reshape(pmes_pitot,(Npitot,1)),
             t_tf_mes_pitot : np.reshape(tmes_pitot,(Npitot,1))
             }

else:
    print('Loading Dense Data')
    x_int,y_int,t_int,s_train,xmes,ymes,tmes,umes,vmes,pmes = ltd.training_dict(Nmes,Nint,Nbc,filename_data,geom,Tintmax=1e2,data_selection = 'all',desync=False, multigrid=multigrid,Ngrid=Ngrid,stdNoise=stdNoise,cut=True,method_int=IntSampling)
    Nmes = len(xmes)
    Tmin = 400.
    
    if multigrid:
        tf_dict = []
        for k in range(Ngrid):
            tf_dict_temp = {x_tf_int : np.reshape(x_int[k],(Nint,1)),
              y_tf_int : np.reshape(y_int[k],(Nint,1)),
              t_tf_int : np.reshape(t_int[k],(Nint,1)),
              s_tf : np.reshape(s_train,(Nbc,1)),
              x_tf_mes : np.reshape(xmes,(Nmes,1)),
              y_tf_mes : np.reshape(ymes,(Nmes,1)),
              p_tf_mes : np.reshape(pmes,(Nmes,1)),
              t_tf_mes : np.reshape(tmes,(Nmes,1)),
              u_tf_mes : np.reshape(umes,(Nmes,1)),
              v_tf_mes : np.reshape(vmes,(Nmes,1))
              }
            tf_dict.append(tf_dict_temp)
        
    else:      
        tf_dict = {x_tf_int : np.reshape(x_int,(Nint,1)),
              y_tf_int : np.reshape(y_int,(Nint,1)),
              t_tf_int : np.reshape(t_int,(Nint,1)),
              s_tf : np.reshape(s_train,(Nbc,1)),
              x_tf_mes : np.reshape(xmes,(Nmes,1)),
              y_tf_mes : np.reshape(ymes,(Nmes,1)),
              p_tf_mes : np.reshape(pmes,(Nmes,1)),
              t_tf_mes : np.reshape(tmes,(Nmes,1)),
              u_tf_mes : np.reshape(umes,(Nmes,1)),
              v_tf_mes : np.reshape(vmes,(Nmes,1))
              }
    

# Validation data set loading
# We extract 10 times more points for both dense measurements and equation penalisation
print('Loading validation data set')

x_int_valid,y_int_valid,t_int_valid,s_train,xmes_valid,ymes_valid,tmes_valid,umes_valid,vmes_valid,pmes_valid = ltd.training_dict(10*Nmes,10*Nint,Nbc,filename_data,geom,Tintmax=1e2,cut=True,method_int='uniform')
Nmesvalid = len(xmes_valid)

tf_dict_valid = {x_tf_int : np.reshape(x_int_valid,(10*Nint,1)),
     y_tf_int : np.reshape(y_int_valid,(10*Nint,1)),
     t_tf_int : np.reshape(t_int_valid,(10*Nint,1)),
     s_tf : np.reshape(s_train,(Nbc,1)),
     x_tf_mes : np.reshape(xmes_valid,(Nmesvalid,1)),
     y_tf_mes : np.reshape(ymes_valid,(Nmesvalid,1)),
     u_tf_mes : np.reshape(umes_valid,(Nmesvalid,1)),
     v_tf_mes : np.reshape(vmes_valid,(Nmesvalid,1)),
     p_tf_mes : np.reshape(pmes_valid,(Nmesvalid,1)),
     t_tf_mes : np.reshape(tmes_valid,(Nmesvalid,1))}


# =============================================================================
# GPU use after loading data
# =============================================================================
print('GPU use after loading data')
GPUtil.showUtilization()


# =============================================================================
# Training
# =============================================================================

nnf.print_bar()
t1 = time.time()
print('Start training after %d s'%(t1-t0))

print('Start L-BFGS-B training')
List_it_loss_LBFGS,List_it_loss_valid_LBFGS = nnf.model_train_scipy(opt_LBFGS,sess,Loss,tf_dict[0],List_loss = True,tf_dict_valid=tf_dict_valid,loss_valid = Loss_dense_mes)

t2 = time.time()
print('L-BFGS-B training ended after %d s'%(t2-t1))

print('Start Adam training')
# Here Adam training is stopped if it reaches a time limit AdamTmax, or number of iterations Nit or if training loss goes under tolAdam
AdamTmax = Tmax-(t2-t0)
List_it_loss_Adam,List_it_loss_valid_Adam = nnf.model_train_Adam(opt_Adam,sess,Loss,liste_tf_dict=tf_dict,Nit=1e5,tolAdam=1e-5,it=it,itdisp=100,maxTime=AdamTmax,multigrid=multigrid,NgridTurn=NgridTurn,List_loss = True,tf_dict_valid=tf_dict_valid,loss_valid = Loss_dense_mes)
t3 = time.time()
print('Adam training ended after %d s'%(t3-t2))

# =============================================================================
# GPU use after training
# =============================================================================
print('GPU use after training')
GPUtil.showUtilization()
print('End of training')

# =============================================================================
# Print residuals errors and losses
# =============================================================================

nnf.print_bar()
print('Error details')
nnf.print_bar()

if not(multigrid):
    tf_dict = [tf_dict]

print('')
nnf.tf_print('Border',loss_BC(s_tf),sess,tf_dict[0])
nnf.tf_print('Loss eqs. modes',Loss_int_mode_wrap,sess,tf_dict[0])
nnf.tf_print('Loss eqs. int time',Loss_int_time_wrap,sess,tf_dict[0])
nnf.tf_print('Loss mesures training',Loss_mes,sess,tf_dict[0])
nnf.tf_print('Loss mesures validation',Loss_dense_mes,sess,tf_dict_valid)
    
if args.DesyncSparseData:
    
    def r_div_eucli(a,b):
        '''
        a,b real numbers
        return r with a = n*b + r, n (int) and -b/2 <= r < b/2
        '''
        rtemp = a%b
        return np.where(rtemp>0.5*b,rtemp-b,rtemp)

    
    print('Validation Resync')
    Delta_phi_tf_pitot_found_o = sess.run(Delta_phi_tf_pitot)
    err_rms_resync = np.sqrt(np.mean(np.square((r_div_eucli(Delta_phi_tf_pitot_found_o-Delta_phi_np_pitot_applied,2*np.pi/omega_0)))))
    err_rms_resync_normalized = err_rms_resync/np.sqrt(np.mean(np.square(Delta_phi_np_pitot_applied)))
    print('Err RMS Resynchro : %.3e'%(err_rms_resync))
    print('Err RMS Resynchro normalized : %.3e'%(err_rms_resync_normalized))
    
    # Plot répartition des  erreurs de resyncro
    xpitot = np.reshape(xmes_pitot,[Ntimes,Nxpitot])[0,:]
    ypitot = np.reshape(ymes_pitot,[Ntimes,Nxpitot])[0,:]
    err_resync_pitot = r_div_eucli(Delta_phi_tf_pitot_found_o-Delta_phi_np_pitot_applied,2*np.pi/omega_0)
    
    size_resync = np.log10(err_resync_pitot)
    
    plt.figure()
    plt.scatter(xpitot,ypitot,c=np.log10(err_resync_pitot),marker='o',s=1.+size_resync)
    plt.colorbar()
    plt.scatter(xmes_cyl,ymes_cyl,c='black',marker='.',s=1.)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('equal')
    plt.xlim((Lxmin,Lxmax))
    plt.ylim((Lymin,Lymax))
    plt.title('Synchronisation error - log')
    plt.tight_layout()
    plt.savefig(repertoire+'/resync_err.png')
    plt.close()
    


# =============================================================================
# Save NN Model coefficients in a pickle archive
# =============================================================================

print('Saving NN Model...')

str_layers_fluid = [str(j) for j in layers]
filename_fluid = repertoire + '/DNN' + '_'.join(str_layers_fluid) + '_tanh.pickle'

Data_fluid = sess.run([w_u,b_u,w_v,b_v,w_p,b_p])
pcklfile_fluide = open(filename_fluid,'ab+')
pickle.dump(Data_fluid,pcklfile_fluide)
pcklfile_fluide.close()
print('Model exported in '+repertoire)

# =============================================================================
# Save convergence history
# =============================================================================

print('Saving convergence history...')

filename_hist = repertoire + '/Convergence_history.pickle'

Data_loss_history = [List_it_loss_LBFGS,List_it_loss_valid_LBFGS,List_it_loss_Adam,List_it_loss_valid_Adam]
pckl_hist = open(filename_hist,'ab+')
pickle.dump(Data_loss_history,pckl_hist)
pckl_hist.close()
print('History exported in '+repertoire)

plt.figure()
plt.scatter(np.array(List_it_loss_LBFGS)[:,0],np.array(List_it_loss_LBFGS)[:,1],label='LBFGS train',marker='.',s=1.,c='red')
# plt.scatter(np.array(List_it_loss_valid_LBFGS)[:,0],np.array(List_it_loss_valid_LBFGS)[:,1],label='LBFGS valid',marker='.',s=1.,c='pink')
# Validation loss does not seem to be accessible during L-BFGS-B training. It returns constant values
plt.scatter(np.array(List_it_loss_Adam)[:,0]+np.max(np.array(List_it_loss_LBFGS)[:,0]),np.array(List_it_loss_Adam)[:,1],label='Adam train',marker='.',s=1.,c='blue')
plt.scatter(np.array(List_it_loss_valid_Adam)[:,0]+np.max(np.array(List_it_loss_LBFGS)[:,0]),np.array(List_it_loss_valid_Adam)[:,1],label='Adam valid',marker='.',s=1.,c='green')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(repertoire+'/Convergence_history.png')
plt.close()



# =============================================================================
# Plot of modal shapes
# =============================================================================


for k in range(Nmodes):
    nnf.tf_plot_scatter_complex(x_tf_int[:,0],y_tf_int[:,0],fluid_u(x_tf_int,y_tf_int)[0,:,k],
                        sess,
                        title='u Mode '+str(k),
                        xlabel='$x$',ylabel='$y$',
                        tf_dict=tf_dict_valid)
    plt.savefig(repertoire+'/u_mode_'+str(k)+'.png')
    plt.close()
    


for k in range(Nmodes):
    nnf.tf_plot_scatter_complex(x_tf_int[:,0],y_tf_int[:,0],fluid_v(x_tf_int,y_tf_int)[0,:,k],
                        sess,
                        title='v Mode '+str(k),
                        xlabel='$x$',ylabel='$y$',
                        tf_dict=tf_dict_valid)
    plt.savefig(repertoire+'/v_mode_'+str(k)+'.png')
    plt.close()

for k in range(Nmodes):
    nnf.tf_plot_scatter_complex(x_tf_int[:,0],y_tf_int[:,0],fluid_p(x_tf_int,y_tf_int)[0,:,k],
                        sess,
                        title='p Mode '+str(k),
                        xlabel='$x$',ylabel='$y$',
                        tf_dict=tf_dict_valid)
    plt.savefig(repertoire+'/p_mode_'+str(k)+'.png')
    plt.close()


# =============================================================================
# Comparison at a given timestep between modalPINN and simulations data
# =============================================================================
inst = 16

Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = ltd.read_cut_simulation_data(filename_data,geom)

tf_dict_compare = {
    x_tf_mes : np.reshape(nodes_X[0,:],(len(nodes_X[0,:]),1)),
    y_tf_mes : np.reshape(nodes_Y[0,:],(len(nodes_Y[0,:]),1)),
    t_tf_mes : np.reshape(times[inst]*np.ones(len(nodes_X[0,:])),(len(nodes_Y[0,:]),1)),
    u_tf_mes : np.reshape(Us[inst,:],(len(nodes_X[0,:]),1))
    }

suptitle='u difference at t = '+'{0:.2f}'.format(times[inst])

nnf.tf_plot_compare_3plot(x_tf_mes,y_tf_mes,u_tf_mes,fluid_u_t(x_tf_mes,y_tf_mes,t_tf_mes),sess,xlabel='$x$',ylabel='$y$',title1='Exact',title2='ModalPINN',suptitle='',tf_dict=tf_dict_compare)
plt.savefig(repertoire+'/diff_u_t_'+'{0:.2f}'.format(times[inst])+'.png')



