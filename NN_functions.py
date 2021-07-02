# -*- coding: utf-8 -*-
"""
This file contains functions specific to
        o neural networks (construction, initialisation),
        o optimisers (calling from scipy or tf interfaces, initialisation, training steps),
        o plots.
@author: Gaétan Raynaud
"""

# =============================================================================
# Libraries
# =============================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import time

# =============================================================================
# Matplotlib parameters
# =============================================================================

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=18)
plt.rc('figure',titlesize=24)


# =============================================================================
# Functions for defining, restoring and initialising neural networks
# =============================================================================

# True --> z = w1*exp(i*w2)
# False --> z = w1 + i*w2
complex_value_exp = True  

def initialize_NN(layers,name_nn=''):        
    '''
    Initialize a complex neural network which structure is defined by layers
    Input layers : list of integers defining the wifth of each layer. 
                   The number of elements in layers defines the depth of the NN
    Return weights : list of matrices filled with tf.complex64 variables
           biases : list of vectors filled with tf.complex64 variables
    '''
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W_1 = xavier_init(size=[layers[l], layers[l+1]],name_w = 'weights_'+name_nn+str(l))
        W_2 = xavier_init(size=[layers[l], layers[l+1]],name_w = 'weights_'+name_nn+str(l))

        b_1 = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name ='biases_'+name_nn+str(l))
        b_2 = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name ='biases_'+name_nn+str(l))
        
        if complex_value_exp:
            W = tf.complex(W_1, 0.)*tf.exp(tf.complex(0., W_2))
            b = tf.complex(b_1, 0.)*tf.exp(tf.complex(0., b_2))
        else :
            W = tf.complex(W_1, W_2)
            b = tf.complex(b_1, b_2)
        
        weights.append(W)
        biases.append(b)     
        
    return weights, biases

def restore_one_NN(layers,w_value,b_value,tf_as_constant=False):
    '''
    input 
    layers : list of integers describing the structure of the NN
    w_value,b_value : list of values of the tensors coefficients
    tf_as_constant : bool (False) : if True, construct directly tf.comple64 coefficients as tf.constant
    If False, construct tf.Variables as tf.float32 and then join them according to complex_value_exp (bool) policy
    ----
    return
    weights and biases tensors variables initialised with the given values
    '''
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        if tf_as_constant:
            W = tf.constant(w_value[l],dtype=tf.complex64,shape=[layers[l],layers[l+1]])
            b = tf.constant(b_value[l],dtype=tf.complex64,shape=[1,layers[l+1]])
        
        else:
            if complex_value_exp:
                W_1 = tf.Variable(tf.math.abs(w_value[l]),dtype=tf.float32,shape=[layers[l],layers[l+1]])
                W_2 = tf.Variable(tf.math.angle(w_value[l]),dtype=tf.float32,shape=[layers[l],layers[l+1]])
                W = tf.complex(W_1, 0.)*tf.exp(tf.complex(0., W_2)) 
                
                b_1 = tf.Variable(tf.math.abs(b_value[l]),dtype=tf.float32,shape=[1,layers[l+1]])
                b_2 = tf.Variable(tf.math.angle(b_value[l]),dtype=tf.float32,shape=[1,layers[l+1]])
                b = tf.complex(b_1, 0.)*tf.exp(tf.complex(0., b_2))
                
            else:
                W_1 = tf.Variable(tf.math.real(w_value[l]),dtype=tf.float32,shape=[layers[l],layers[l+1]])
                W_2 = tf.Variable(tf.math.imag(w_value[l]),dtype=tf.float32,shape=[layers[l],layers[l+1]])
                W = tf.complex(W_1,W_2)
    
                b_1 = tf.Variable(tf.math.real(b_value[l]),dtype=tf.float32,shape=[1,layers[l+1]])
                b_2 = tf.Variable(tf.math.imag(b_value[l]),dtype=tf.float32,shape=[1,layers[l+1]])
                b = tf.complex(b_1,b_2)
        weights.append(W)
        biases.append(b)
    return weights,biases
            

def restore_NN(layers,filename_restore,tf_as_constant=False):

    '''
    Restore u, v and p ModalPINN models 
    Input layers : list of each layers width
          filename_restore (str) : location of the pickle archive where the values are stored
          tf_as_constant (bool) : if True, model's parameters are initialised as tf.constant 
                                  (and are therefore fixed). Else, they are set as 
                                  tf.variable and can be trained once again.
    Return the weights and biases 
    '''    

    file = open(filename_restore,'rb')
    w_u_value,b_u_value,w_v_value,b_v_value,w_p_value,b_p_value = pickle.load(file)
    file.close()

    w_u,b_u = restore_one_NN(layers,w_u_value,b_u_value,tf_as_constant)
    w_v,b_v = restore_one_NN(layers,w_v_value,b_v_value,tf_as_constant)
    w_p,b_p = restore_one_NN(layers,w_p_value,b_p_value,tf_as_constant)
    
    return w_u,b_u,w_v,b_v,w_p,b_p
        
        

def xavier_init(size,name_w):
    '''
    Initialisation of weights using xavier init.
    This function comes from Raissi et al. (2019)
    '''
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32, name = name_w)


def neural_net(X, weights, biases):
    '''
    Construct one neural network as a succession of affine transformation and 
    non-linear functions (here sigma = tanh)
    Input : tf.tensor X, weights and biases that define the model
    Output: tf.tensor Y
    '''
    H = X
    num_layers = len(weights) + 1
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


def f_BC5(x, y, geom, fact=5.):
    '''
    return real tensor of same size than x and y
    equal to zero on the cylinder border
    '''
    Lxmin,Lxmax,Lymin,Lymax,x_c,y_c,r_c = geom
    r = tf.sqrt(tf.square(x-x_c) + tf.square(y-y_c)) - r_c
    return tf.tanh(fact*r)

def out_nn_modes_uv(x,y,weights,biases,geom):
    '''
    Return Nmode complex modes shapes of DNN defined with weights and biases
    Prior dictionary f_BC5 is applied so that each mode shape verifies =0 on cylinder's border
    Input x,y : [Nint,1] tf.float32 tensor
    Output shape : [1,Nint,Nmode] tf.complex64 tensor
    '''
    xint = tf.complex(x,0.)
    yint = tf.complex(y,0.)
    out_nn = neural_net(tf.transpose(tf.stack([xint,yint])),weights,biases)
    Nmode = int(out_nn[0,0,:].shape[0])
    fbc5c = tf.complex(f_BC5(x,y,geom)[:,0],0.)
    t_parts = tf.convert_to_tensor([fbc5c*out_nn[:,:,k] for k in range(Nmode)])
    return tf.transpose(t_parts,perm=[1,2,0])

def out_nn_modes_p(x,y,weights,biases):
    '''
    Return Nm complex modes of dnn defined with weights and biases
    Input x,y : [Nint,1] real tf.float32 tensor
    Output shape : [1,Nint,Nmode] tf.complex64 tensor
    '''
    xint = tf.complex(x,0.)
    yint = tf.complex(y,0.)
    out_nn = neural_net(tf.transpose(tf.stack([xint,yint])),weights,biases)
    Nmode = int(out_nn[0,0,:].shape[0])
    t_parts = tf.convert_to_tensor([out_nn[:,:,k] for k in range(Nmode)])
    return tf.transpose(t_parts,perm=[1,2,0])


def NN_time_uv(x,y,t,weights,biases,geom,omega_0,trunc_mode=None):
    '''
    x,y,t : [Nint,1] tf.float32 tensors, list of coordinates (x,t) where to compute u or v(x,y,t)
    omega_0 : fondamental frequency
    Output [Nint,1] tf.float32 tensor
    trunc_mode : int (or None) : if an integer value is provided, select only 
                the trunc_mode first mode given. Else if trunc_mode=None, use all modes
    '''
    out_NN = out_nn_modes_uv(x,y,weights,biases,geom) 
    Nmode = int(out_NN[0,0,:].shape[0])
    if trunc_mode!=None and trunc_mode <= Nmode:
        Nmode = trunc_mode
    parts = [out_NN[0,:,k]*tf.exp(k*omega_0*tf.complex(0.,t[:,0])) for k in range(Nmode)]
    t_parts = tf.convert_to_tensor(parts)
    t_real = tf.real(tf.reduce_sum(t_parts,axis=0))
    return tf.transpose(tf.convert_to_tensor([t_real])) # retrait de perm=[1,0]

def NN_time_p(x,y,t,weights,biases,omega_0,trunc_mode=None):
    '''
    x,y,t : [Nint,1] tf.float32 tensors, list of coordinates (x,t) where to compute p(x,y,t)
    omega_0 : fondamental frequency
    Output [Nint,1] tf.float32 tensor
    '''
    out_NN = out_nn_modes_p(x,y,weights,biases) 
    Nmode = int(out_NN[0,0,:].shape[0])
    if trunc_mode!=None and trunc_mode <= Nmode:
        Nmode = trunc_mode
    parts = [out_NN[0,:,k]*tf.exp(k*omega_0*tf.complex(0.,t[:,0])) for k in range(Nmode)]
    t_parts = tf.convert_to_tensor(parts)
    t_real = tf.real(tf.reduce_sum(t_parts,axis=0))
    return tf.transpose(tf.convert_to_tensor([t_real]))


# =============================================================================
# Declaration of the optimisers
# =============================================================================

def declare_LBFGS(loss,maxit=50000,maxfun=50000,ftol=1.0 * np.finfo(float).eps):
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method = 'L-BFGS-B', 
                                                                options = {'maxiter': maxit, #50000
                                                                           'maxfun': maxfun, #50000
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : ftol}) 
    print('L-BFGS-B optimizer declared with maxit = %d, maxfun = %d, ftol = %.2e'%(maxit,maxfun,ftol))
    return optimizer


def declare_Adam(loss,lr=1e-3,*args): #list_var=tf.trainable_variables()
    '''
    *args :
        var_list : trainable variables
    '''
    optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    print('Adam optimize declared with learning rate = %.2e'%(lr))
    if len(args) == 1:
        list_var = args[0]
        return optimizer_Adam.minimize(loss,var_list=list_var)
    else:
        return optimizer_Adam.minimize(loss)


def declare_init_session():
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    return sess

def square_norm(z):
    '''
    z : tf.complex64 tensor
    return tf tensor of square norm value of each complex
    '''
    return tf.square(tf.math.real(z)) + tf.square(tf.math.imag(z))

def square_norm_np(z):
    '''
    z : numpy array of complex values
    return np array of same dimension with square norm value of each complex number
    '''
    return np.square(np.real(z)) + np.square(np.imag(z))


# =============================================================================
# Entrainement
# =============================================================================


def simple_callback(loss):
    print('Loss: %.3e' % (loss))



def model_train_scipy(optimizer,sess,loss,tf_dict=None,fn_callback=simple_callback,List_loss = False, loss_valid = None, tf_dict_valid=None):
    global it
    it = 0
    global List_it_loss_LBFGS
    List_it_loss_LBFGS = []
    global List_it_loss_valid_LBFGS
    List_it_loss_valid_LBFGS = []
    
    if List_loss == True and tf_dict_valid != None:
    
        def callback(loss,List_loss=List_loss):
            global it
            global List_it_loss_LBFGS
            global List_it_loss_valid_LBFGS
            it += 1
            List_it_loss_LBFGS.append([it,loss])
            if List_loss==True and it%100 == 0:
                loss_valid_value = sess.run(loss_valid,feed_dict=tf_dict_valid)
                List_it_loss_valid_LBFGS.append([it,loss_valid_value])
            print('Loss: %.3e' % (loss))
            # global it
            # it += 1
        
        fn_callback=callback
    
    if tf_dict != None:
        optimizer.minimize(sess,
                feed_dict = tf_dict,
                fetches = [loss],
                loss_callback = fn_callback)
    else:
        optimizer.minimize(sess,
                fetches = [loss],
                loss_callback = fn_callback) 
    return List_it_loss_LBFGS,List_it_loss_valid_LBFGS

def model_train_Adam(optimizer,sess,loss,liste_tf_dict=None,Nit=1e4,tolAdam=1e-4,it=0,itdisp=1000,maxTime=None,multigrid=False,NgridTurn=1000,List_loss = False, loss_valid = None, tf_dict_valid=None):
    List_it_loss_Adam = []
    List_it_loss_valid_Adam = []
    if not(multigrid):
        tf_dict=[liste_tf_dict]
        NgridTurn=1
        Ngrid = 1
    else:
        tf_dict = liste_tf_dict
        Ngrid = len(tf_dict)
        
    t0 = time.time()
    loss_value = sess.run(loss, tf_dict[0])
    it0 = it
    conditionTime = True
    while(it-it0<Nit and loss_value>tolAdam and conditionTime):
        
        k_dict = int(it/NgridTurn)%Ngrid
        
        sess.run(optimizer, tf_dict[k_dict])
        loss_value = sess.run(loss, tf_dict[k_dict])
        
        if it%itdisp ==0:
            print('Post Adam it %d - Loss value :  %.3e' % (it, loss_value))
            
        if List_loss==True and it%100 == 0:
            loss_valid_value = sess.run(loss_valid,feed_dict=tf_dict_valid)
            List_it_loss_valid_Adam.append([it,loss_valid_value])
        List_it_loss_Adam.append([it,loss_value])
        
        
        it += 1
        conditionTime = (maxTime==None) or ((time.time()-t0)<maxTime)
        
    return List_it_loss_Adam,List_it_loss_valid_Adam




# =============================================================================
# Print and plot
# =============================================================================

def print_bar():
    print('--------------------------------------------')

def tf_print(string,tensor,sess,tf_dict=None):
    '''
    Parameters
    ----------
    string : String of character to display before the result of tf output
    tensor : tensor to compute and print
    sess : Current session object to compute given tensor
    tf_dict : dictionnary to feed, in cas it is necessary
    '''
    print(string + " " + str(sess.run(tensor,feed_dict=tf_dict)))
    


def tf_plot_scatter(x_tf,y_tf,c_tf,sess,xlabel='x',ylabel='y',title='',tf_dict=None):
    '''
    Parameters
    ----------
    x_tf, y_tf, c_tf : 1D tensor to compute and plot
    sess : sess object to compute given tensors
    xlabel, ylabel, title : string to display on figure
    tf_dict = None : dictionnary to provide for computation of given tensors
    -------
    Returns pyplot fig object and ax
    '''
    # Step 1 : copputation
    x_np,y_np,c_np = sess.run([x_tf,y_tf,c_tf],feed_dict=tf_dict)
    
    # Step 2 : plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x_np,y_np,c=c_np,marker='.',s=1.)
    ax.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    fig.tight_layout()
    
    return fig,ax


def tf_plot_scatter_complex_4fig(x_tf,y_tf,c_tf,sess,xlabel='x',ylabel='y',title='',tf_dict=None):
    '''
    Parameters
    ----------
    x_tf, y_tf, c_tf : 1D tensor to compute and plot
    x and y are float32 tensors
    c is complex64 tensor
    sess : sess object to compute given tensors
    xlabel, ylabel, title : string to display on figure
    tf_dict = None : dictionnary to provide for computation of given tensors
    -------
    Returns pyplot fig object and ax
    '''
    # Step 1 : copputation
    x_np,y_np,c_np = sess.run([x_tf,y_tf,c_tf],feed_dict=tf_dict)
    
    # Step 2 : plot
    fig = plt.figure(figsize=(8,6))
    plt.subplot(221)
    plt.scatter(x_np,y_np,c=np.real(c_np),marker='.',s=1.)
    plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+' - real part')
    plt.colorbar()
    plt.subplot(222)
    plt.scatter(x_np,y_np,c=np.imag(c_np),marker='.',s=1.)
    plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+ ' - imaginary part')
    plt.colorbar()
    plt.subplot(223)
    plt.scatter(x_np,y_np,c=np.sqrt(np.square(np.real(c_np))+np.square(np.imag(c_np))),marker='.',s=1.)
    plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+' - norm')
    plt.colorbar()
    plt.subplot(224)
    plt.scatter(x_np,y_np,c=np.angle(c_np),marker='.',s=1.)
    plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+ ' - angle')
    plt.colorbar()
    plt.tight_layout()
    
    return fig


def tf_plot_scatter_complex(x_tf,y_tf,c_tf,sess,xlabel='x',ylabel='y',title='',tf_dict=None):
    '''
    Parameters
    ----------
    x_tf, y_tf, c_tf : 1D tensor to compute and plot
    x and y are float32 tensors
    c is complex64 tensor
    sess : sess object to compute given tensors
    xlabel, ylabel, title : string to display on figure
    tf_dict = None : dictionnary to provide for computation of given tensors
    -------
    Returns pyplot fig object and ax
    '''
    # Step 1 : copputation
    x_np,y_np,c_np = sess.run([x_tf,y_tf,c_tf],feed_dict=tf_dict)
    
    # Step 2 : plot
    fig = plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.scatter(x_np,y_np,c=np.real(c_np),marker='.',s=1.)
    plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+' - real part')
    plt.colorbar()
    plt.subplot(122)
    plt.scatter(x_np,y_np,c=np.imag(c_np),marker='.',s=1.)
    plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+ ' - imaginary part')
    plt.colorbar()
    plt.tight_layout()
    
    return fig


def tf_plot_compare_3plot(x_tf,y_tf,c_tf_1,c_tf_2,sess,xlabel='x',ylabel='y',title1='',title2='',suptitle='',tf_dict=None):
    '''
    x_tf, y_tf, c_tf_1, c_tf_2 : 1D float32 tensor to compute and plot
    Return a figure with 3 subplots : c_tf_1, c_tf_2, log10((c_tf_1-c_tf_2)^2)
    '''
    
    x_np,y_np,c_np_1,c_np_2 = sess.run([x_tf,y_tf,c_tf_1,c_tf_2],feed_dict=tf_dict)
    
    fig = plt.figure(figsize=(15,4))
    plt.subplot(131)
    plt.scatter(x_np,y_np,c=c_np_1,marker='.',s=1.)
    plt.axis('equal')
    plt.colorbar()
    plt.title(title1)
    plt.subplot(132)
    plt.scatter(x_np,y_np,c=c_np_2,marker='.',s=1.)
    plt.axis('equal')
    plt.colorbar()
    plt.title(title2)
    plt.subplot(133)
    plt.scatter(x_np,y_np,c=np.log10(np.square(c_np_1-c_np_2)),marker='.',s=1.)
    plt.axis('equal')
    plt.colorbar()
    plt.title('Square difference - log10')
    plt.suptitle(suptitle)
    plt.tight_layout()
    
    return fig