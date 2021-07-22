# -*- coding: utf-8 -*-
"""
PINN Python Code
This is the main Python file for performing flow reconstruction using classical PINN
for comparison with ModalPINN. See details in the readme.md or in the main file ModalPINN_VortexShedding.py

@author: Gaétan Raynaud. 
ORCID : orcid.org/0000-0002-2802-7366
email : gaetan.raynaud (at) polymtl.ca
"""
# =============================================================================
# Import des bibliotheques
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
import NN_functions_classicPINN as nnf
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
repertoire = 'OutputPythonScript/ClassicPINN_'+ d.strftime("%Y_%m_%d-%H_%M_%S") + '__' +str(r)
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
parser.add_argument('--Nmes',type=int,default=5000,help="Number of measurement points to provide for optimisation")
parser.add_argument('--Nint',type=int,default=50000,help="Number of computing points to provide for equation evaluation during optimisation")
parser.add_argument('--multigrid',action="store_true",default=False,help="Use of multi grid")
parser.add_argument('--Ngrid',type=int,default=1,help="Number of batch for Adam optimization")
parser.add_argument('--NgridTurn',type=int,default=1000,help="Number of iterations between each batch changement")
parser.add_argument('--Noise',type=float,default=0.,help="Define standard deviation of gaussian noise added to measurements")
parser.add_argument('--WidthLayer',type=int,default=20.,help="Number of neurons per layer and per mode")
parser.add_argument('--DesyncSparseData',action="store_true",default=False,help="if activated (and --SparseData == True), then simulated measurements are randomly made out of synchronisation")

args = parser.parse_args()

print('Args passed to python script')
print('Tmax '+str(args.Tmax)+' (h)')
print('Nmes %d' % (args.Nmes))
print('Nint %d' % (args.Nint))
print('Multigrid : '+str(args.multigrid))
print('Ngrid : '+str(args.Ngrid))
print('Ngrid Turn : '+str(args.NgridTurn))
print('STD Noise : %.2e' % (args.Noise))
print('Neurons per layer : %.2e' % (args.WidthLayer))

# =============================================================================
# Physical and geometrical parameters 
# =============================================================================

Re = 100.
Lxmin = -4. #-40.
Lxmax = 8. #120.
Lx = Lxmax-Lxmin
Lymin = -4. #-60.
Lymax = 4. #60.
Ly = Lymax-Lymin
x_c = 0.
y_c = 0.
r_c = 0.5
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


Nmes = args.Nmes
Nint = args.Nint
Nbc = 1000
multigrid = args.multigrid
Ngrid = args.Ngrid
NgridTurn = args.NgridTurn
stdNoise = args.Noise

# Structure of each Neural Network that approximate u,v or p as a function of x,y and t
layers = [3,args.WidthLayer,args.WidthLayer,args.WidthLayer,args.WidthLayer,1]

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


# =============================================================================
# Model construction
# =============================================================================

w_u,b_u = nnf.initialize_NN(layers)
w_v,b_v = nnf.initialize_NN(layers)
w_p,b_p = nnf.initialize_NN(layers)

def fluid_u_t(x,y,t):
    '''
    Compute u at instant t and position x,y
    Input: x,y,t TF tensors of shape [Nint,1]
    Return TF tensor of shape [Nint,1] with real values
    '''
    return nnf.NN_time_uv(x,y,t,w_u,b_u,geom)

def fluid_v_t(x,y,t):
    '''
    Compute v at instant t and position x,y
    Input: x,y,t TF tensors of shape [Nint,1]
    Return TF tensor of shape [Nint,1] with real values
    '''
    return nnf.NN_time_uv(x,y,t,w_v,b_v,geom)

def fluid_p_t(x,y,t):
    '''
    Compute p at instant t and position x,y
    Input: x,y,t TF tensors of shape [Nint,1]
    Return TF tensor of shape [Nint,1] with real values
    '''
    return nnf.NN_time_p(x,y,t,w_p,b_p)

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
    Ns = 1000
    s_cyl = tf.random.uniform([Ns,1], minval=0., maxval = 1., dtype = tf.float32)*tf.transpose(1+0.*t)
    s_cyl_r = tf.reshape(s_cyl,[Nt*Ns,1])
    x_cyl_r = tf.reshape(xbc5(s_cyl_r),[Nt*Ns,1])
    y_cyl_r = tf.reshape(ybc5(s_cyl_r),[Nt*Ns,1])
    t_cyl = (1.+0*s_cyl)*tf.transpose(t)
    t_cyl_r = tf.reshape(t_cyl,[Nt*Ns,1])
    
    u = fluid_u_t(x_cyl_r,y_cyl_r,t_cyl_r)
    v = fluid_v_t(x_cyl_r,y_cyl_r,t_cyl_r)
    p = fluid_p_t(x_cyl_r,y_cyl_r,t_cyl_r)
    
    u_x = tf.gradients(u, x_cyl_r)[0]
    u_y = tf.gradients(u, y_cyl_r)[0]
    u_xx = tf.gradients(u_x, x_cyl_r)[0]
    u_yy = tf.gradients(u_y, y_cyl_r)[0]
    
    v_x = tf.gradients(v, x_cyl_r)[0]
    v_y = tf.gradients(v, y_cyl_r)[0]
    v_xx = tf.gradients(v_x, x_cyl_r)[0]
    v_yy = tf.gradients(v_y, y_cyl_r)[0]
    
    
    nx_base = - tf.gradients(y_cyl_r, s_cyl_r)[0]
    ny_base = tf.gradients(x_cyl_r, s_cyl_r)[0]
    normalisation = tf.sqrt(tf.square(nx_base) + tf.square(ny_base))
    nx = nx_base/normalisation
    ny = ny_base/normalisation
    
    fx_tf_local = -p*nx + 2.*(1./Re)*u_x*nx + (1./Re)*(u_y+v_x)*ny
    fy_tf_local = -p*ny + 2.*(1./Re)*v_y*ny + (1./Re)*(u_y+v_x)*nx
    
    # Reshape en [Ns,Nt]
    fx_tf_local_r2 = tf.reshape(fx_tf_local,[Ns,Nt])
    fy_tf_local_r2 = tf.reshape(fy_tf_local,[Ns,Nt])
    
    fx_tf = -2.*np.pi*r_c*tf.reduce_mean(fx_tf_local_r2,axis=0)
    fy_tf = -2.*np.pi*r_c*tf.reduce_mean(fy_tf_local_r2,axis=0)
    
    return fx_tf,fy_tf


# =============================================================================
# Definition of functions for loss
# =============================================================================

def loss_int_time(x,y,t):
    '''
    Parameters
    ----------
    x,y,t : float 32 tensor [Nint,1]

    Returns
    -------
    Return [Nint,1] tensor containing square error on NS equations
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
    Return [Nmes,1] tf.float32 tensor containing square difference to measurements 
    '''
    p_DNN = fluid_p_t(xmes,ymes,tmes)
    
    return tf.square(p_DNN-pmes)


def loss_BC(s,t):
    '''
    Return erreur on u=v=0 on cylinder border for each mode
    Input s : [Nbc,1] tf.float32 tensor of coordinates \in [0,1]
    Output : [] tf.float32 real positive number
    '''    
    x = xbc5(s)
    y = ybc5(s)
    u = fluid_u_t(x,y,t)
    v = fluid_v_t(x,y,t)
    
    err = tf.square(u) + tf.square(v)
    
    return tf.reduce_sum(tf.reduce_mean(err))


# =============================================================================
# Training loss creation
# =============================================================================

# Wrap error on (u,v,p) measurements
Loss_mes = tf.reduce_mean(loss_mes(x_tf_mes,y_tf_mes,t_tf_mes,u_tf_mes,v_tf_mes,p_tf_mes))

# Wrap error on physical equations
Loss_int_time_wrap = tf.reduce_mean(loss_int_time(x_tf_int, y_tf_int ,t_tf_int))

# Wrap error on (u,v) measurements at simulated pitot probes locations
Loss_mes_pitot = tf.reduce_mean(loss_mes_uv(x_tf_mes_pitot,y_tf_mes_pitot,t_tf_mes_pitot_resync,u_tf_mes_pitot,v_tf_mes_pitot))
Loss_mes_pitot_desync = tf.reduce_mean(loss_mes_uv(x_tf_mes_pitot,y_tf_mes_pitot,t_tf_mes_pitot,u_tf_mes_pitot,v_tf_mes_pitot))

# Wrap error on pressure measurement around cylindre border
Loss_mes_cyl = tf.reduce_mean(loss_mes_p(x_tf_mes_cyl,y_tf_mes_cyl,t_tf_mes_cyl,p_tf_mes_cyl))

# Simulated experimental losses
Loss_mes_exp =  Loss_mes_pitot + Loss_mes_cyl


Loss = Loss_int_time_wrap + Loss_mes

# =============================================================================
# Optimizer configuration
# =============================================================================

opt_LBFGS = nnf.declare_LBFGS(Loss,ftol=1e-7)

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

x_int,y_int,t_int,s_train,xmes,ymes,tmes,umes,vmes,pmes = ltd.training_dict(Nmes,Nint,Nbc,filename_data,geom,Tintmin=400.,Tintmax=420.,data_selection = 'all',desync=False, multigrid=multigrid,Ngrid=Ngrid,stdNoise=stdNoise,cut=True,method_int='uniform')
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



x_int_valid,y_int_valid,t_int_valid,s_train,xmes_valid,ymes_valid,tmes_valid,umes_valid,vmes_valid,pmes_valid = ltd.training_dict(10*Nmes,10*Nint,Nbc,filename_data,geom,Tintmax=1e2,cut=True)
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
List_it_loss_LBFGS,List_it_loss_valid_LBFGS = nnf.model_train_scipy(opt_LBFGS,sess,Loss,tf_dict[0],List_loss = True,tf_dict_valid=tf_dict_valid,loss_valid = Loss_mes)

t2 = time.time()
print('L-BFGS-B training ended after %d s'%(t2-t1))

print('Start Adam training')
# Here Adam training is stopped if it reaches a time limit AdamTmax, or number of iterations Nit or if training loss goes under tolAdam
AdamTmax = Tmax-(t2-t0)
List_it_loss_Adam,List_it_loss_valid_Adam = nnf.model_train_Adam(opt_Adam,sess,Loss,liste_tf_dict=tf_dict,Nit=1e5,tolAdam=1e-5,it=it,itdisp=100,maxTime=AdamTmax,multigrid=multigrid,NgridTurn=NgridTurn,List_loss = True,tf_dict_valid=tf_dict_valid,loss_valid = Loss_mes)
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
nnf.tf_print('Loss eqs. int time',Loss_int_time_wrap,sess,tf_dict[0])
nnf.tf_print('Loss mesures training',Loss_mes,sess,tf_dict[0])
nnf.tf_print('Loss mesures validation',Loss_mes,sess,tf_dict_valid)


# =============================================================================
# Save NN Model coefficients in a pickle archive
# =============================================================================

print('Saving NN Model...')

str_layers_fluid = [str(j) for j in layers]
filename_fluid = repertoire + '/DNN' + '_'.join(str_layers_fluid) + '_sine.pickle'

Data_fluid = sess.run([w_u,b_u,w_v,b_v,w_p,b_p])
pcklfile_fluide = open(filename_fluid,'ab+')
pickle.dump(Data_fluid,pcklfile_fluide)
pcklfile_fluide.close()
print('Model exported in '+repertoire)


# =============================================================================
# Comparison at a given timestep between Classic PINN and simulations data
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

