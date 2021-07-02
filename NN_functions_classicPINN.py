# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:05:35 2020

@author: garaya
"""

# =============================================================================
# Import des bibliotheques
# =============================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import time

# =============================================================================
# Param matplotlib
# =============================================================================

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes',titlesize=20)
plt.rc('legend',fontsize=18)
plt.rc('figure',titlesize=24)


# =============================================================================
# Fonctions pour définir le NN
# =============================================================================


def initialize_NN(layers,name_nn=''):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]],name_w = 'weights_'+name_nn+str(l))
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name ='biases_'+name_nn+str(l))
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
                W = tf.Variable(w_value[l],dtype=tf.float32,shape=[layers[l],layers[l+1]])
                b = tf.Variable(b_value[l],dtype=tf.float32,shape=[1,layers[l+1]])
        weights.append(W)
        biases.append(b)
    return weights,biases
            

def restore_NN(layers,filename_restore,tf_as_constant=False):

    file = open(filename_restore,'rb')
    w_u_value,b_u_value,w_v_value,b_v_value,w_p_value,b_p_value = pickle.load(file)
    file.close()

    w_u,b_u = restore_one_NN(layers,w_u_value,b_u_value,tf_as_constant)
    w_v,b_v = restore_one_NN(layers,w_v_value,b_v_value,tf_as_constant)
    w_p,b_p = restore_one_NN(layers,w_p_value,b_p_value,tf_as_constant)
    
    return w_u,b_u,w_v,b_v,w_p,b_p
        
        

def xavier_init(size,name_w):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32, name = name_w)


def neural_net(X, weights, biases):
    H = X
    num_layers = len(weights) + 1
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.sin(tf.add(tf.matmul(H, W), b))
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
    #r = tf.sqrt(tf.abs(tf.square(x-x_c) + tf.square(y-y_c) - tf.square(r_c + 0*x)))
    r = tf.sqrt(tf.square(x-x_c) + tf.square(y-y_c)) - r_c
    return tf.tanh(fact*r)



def NN_time_uv(x,y,t,weights,biases,geom,omega_0,trunc_mode=None):
    '''
    x,y,t : [Nint,1] tf.float32 tensors, list of coordinates (x,t) where to compute u or v(x,y,t)
    omega_0 : fondamental frequency
    output : [Nint,1]
    '''
    out_nn = neural_net(tf.transpose(tf.stack([x,y,t])),weights,biases)
    return tf.transpose(out_nn[:,:,0]*f_BC5(x,y,geom)[:,0])

def NN_time_p(x,y,t,weights,biases,omega_0,trunc_mode=None):
    '''
    x,y,t : [Nint,1] tf.float32 tensors, list of coordinates (x,t) where to compute p(x,y,t)
    output : [Nint,1]
    '''
    out_nn = neural_net(tf.transpose(tf.stack([x,y,t])),weights,biases)
    return tf.transpose(out_nn[:,:,0])



# =============================================================================
# Declaration optimiseurs
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
    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    return sess

def square_norm(z):
    '''
    z : tf.complex64 tensor
    return tensor of square norm values of each complex
    '''
    return tf.square(tf.math.real(z)) + tf.square(tf.math.imag(z))

def square_norm_np(z):
    '''
    z : numpy array of complex values
    return array of same dimension with square norm value of each complex number
    '''
    return np.square(np.real(z)) + np.square(np.imag(z))


# =============================================================================
# Entrainement
# =============================================================================


def callback(loss):
    print('Loss: %.3e' % (loss))
    # global it
    # it += 1


def model_train_scipy(optimizer,sess,loss,tf_dict=None,fn_callback=callback):
    if tf_dict != None:
        optimizer.minimize(sess,
                feed_dict = tf_dict,
                fetches = [loss],
                loss_callback = callback)
    else:
        optimizer.minimize(sess,
                fetches = [loss],
                loss_callback = callback) 

def model_train_Adam(optimizer,sess,loss,liste_tf_dict=None,Nit=1e4,tolAdam=1e-4,it=0,itdisp=1000,maxTime=None,multigrid=False,NgridTurn=1000):
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
        it += 1
        conditionTime = (maxTime==None) or ((time.time()-t0)<maxTime)





# =============================================================================
# Print et plot
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