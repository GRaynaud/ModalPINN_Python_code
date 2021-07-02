"""
Author: Mouad Boudina
From: https://zenodo.org/record/5039610
"""
import numpy as np

from scipy.optimize  import curve_fit
from scipy.integrate import trapz
#==============================================================================
labels_solid = {1:r'$X/D$',
                2:r'$Y/D$',
                3:r'$\theta$',
                4:r'$U/U_{0}$',
                5:r'$V/U_{0}$',
                6:r'$\dot{\theta}$',
                7:r'$F_{X}$',
                8:r'$F_{Y}$',
                9:r'$M_{z}$'}

labels_entity = {1:r'$F_{X}$',
                 2:r'$F_{Y}$',
                 3:r'$M_{z}$'}

labels = {'solid':labels_solid, 'entity':labels_entity}

#fontsize = 12
fontsize = 10

tol = 1e-2

def floatIt(l):
    return np.array([float(e) for e in l])

def extract_reactions(infile):
    f = open(infile, 'r')

    # A dummy test to know whether we are reading reactions of an entity or a
    # solid (in 2D).
    line = f.readline()
    if len(line.strip().split()) == 5:
        flag = 'entity'
    else:
        flag = 'solid'
    f.seek(0)

    times = []

    if flag == 'entity':
        Fx, Fy, Mz = [], [], []

        for line in f:
            # The ':-1' is used to avoid floating the name of the entity in the
            # last column.
            fragmented = floatIt(line.strip().split()[:-1])

            times.append(fragmented[0])

            Fx.append(fragmented[1])
            Fy.append(fragmented[2])
            Mz.append(fragmented[3])

        return np.array(times), np.array(Fx), np.array(Fy), np.array(Mz), flag

    else:
        X, Y, theta = [], [], []
        U, V, theta_dot = [], [], []
        Fx, Fy, Mz = [], [], []

        for line in f:
            # The ':-1' is used to avoid floating the name of the solid in the
            # last column.
            fragmented = floatIt(line.strip().split()[:-1])

            times.append(fragmented[0])

            X.append(fragmented[1])
            Y.append(fragmented[2])
            theta.append(fragmented[3])

            U.append(fragmented[4])
            V.append(fragmented[5])
            theta_dot.append(fragmented[6])

            Fx.append(fragmented[7])
            Fy.append(fragmented[8])
            Mz.append(fragmented[9])

        return np.array(times), \
               np.array(X), np.array(Y), np.array(theta), \
               np.array(U), np.array(V), np.array(theta_dot), \
               np.array(Fx), np.array(Fy), np.array(Mz), flag

def plot_reactions(reactions, variable_index, ax, color):
    plot_params = {'linestyle':'-',
                   'color'    :color,
                   'marker'   :'.',
                   'markerfacecolor':'cyan',
                   'label'    :labels[reactions[-1]][variable_index]}

    ax.plot(reactions[0], reactions[variable_index], **plot_params)

    ax.set_xlabel(r'$\bar{t}=tU_{0}/D$', fontsize=12)
    ax.set_ylabel(r'$F_{y}$', fontsize=12)

#    ax.legend(loc='best', numpoints=1,
#              fontsize=fontsize,
#              frameon=False,
#              ncol=1,
#              labelspacing=0.2,
#              handlelength=0.2)

def get_Fd_and_Fl(reactions, alpha):
    if reactions[-1] == 'solid':
        exception_msg = 'We are sorry, but this function is useful only for' \
                      + 'fixed objects.'
        raise Exception(exception_msg)

    # WARNING:
    # alpha IS IN DEGREES. MUST BE CONVERTED TO RADIANS TO USE IT IN PYTHON FUNCTIONS.
    a = (np.pi/180.)*alpha

    times, Fx, Fy = reactions[:3]

    Fd =  Fx*np.cos(a) + Fy*np.sin(a)
    Fl = -Fx*np.sin(a) + Fy*np.cos(a)

    return times, Fd, Fl

def first_maximum(reactions, variable_index, i0):
    """
    Find the first occurrence of the maximum, starting from index i0, and
    returns that index with the value of the maximum.
    """
    times = reactions[0]

    psi = reactions[variable_index]

    Nt = len(times)
    for i in range(i0, Nt-1):
        if psi[i] > max(psi[i-1], psi[i+1]):
            return psi[i], i

    raise Exception('The function is probably constant, or hasn\'t the same peak.')

def find_t400_t410(reactions):
    times = reactions[0]
    Nt = len(times)

    i400, i410 = 0, 0

    for i in range(Nt):
        if times[i] > 400.:
            i400 = i
            break

    for i in range(i400+1, Nt):
        if times[i] > 410.:
            i410 = i
            break

    return i400, i410

def find_period(reactions, variable_index, ax):
#    if reactions[-1] == 'entity':
#        exception_msg = 'We are sorry, but this function is useful only for' \
#                      + ' moving objects.'
#        raise Exception(exception_msg)

    times = reactions[0]

    i400, i410 = find_t400_t410(reactions)

    psi_max, imax = first_maximum(reactions, variable_index, i400)

    psi_max2, imax2 = first_maximum(reactions, variable_index, imax+1)
    # Sometimes there is local maximums, so we keep searching until we find
    # the real maximum that has the same value as psi_max.
    while abs(psi_max - psi_max2) > tol:
        psi_max2, imax2 = first_maximum(reactions, variable_index, imax2+1)

    psi = reactions[variable_index]

    if ax != None:
        ax.plot(times[i400:i410], psi[i400:i410], color='blue', linestyle='-')

        ax.plot(times[[imax, imax2]], psi[[imax, imax2]],
                color='red', linestyle='-', marker='o')

        ax.set_xlabel('Time', size='xx-large')
        ax.set_ylabel(labels[reactions[-1]][variable_index], size='xx-large')

    period = times[imax2] - times[imax]
    print('FLOW PERIOD = %0.6f' % period)

    mean = np.mean(psi[imax:imax2])
    print('MEAN = %0.6f' % mean)

    maxi = max(psi[imax:imax2])
    print('MAX  = %0.6f' % maxi)

    amp = maxi - mean
    print('AMP  = %0.6f' % amp)

    return period, mean, maxi, amp

def fit_a_sine(reactions, variable_index, ax):
    period, mean, maxi, amp = find_period(reactions, variable_index, ax)

    times = reactions[0]
    psi = reactions[variable_index]
    cent_norm = (psi - mean)/amp

    i400, i410 = find_t400_t410(reactions)

    def f(t, phi):
        return np.sin(2*np.pi*t/period + phi)

    popt, pcov = curve_fit(f, times[i400:i410], cent_norm[i400:i410])
    phi = popt

    print('PHASE LAG = %0.6f = %0.3f PI' % (phi, phi/np.pi))

    if ax != None:
        ax.plot(times[i400:i410], mean + amp*f(times[i400:i410], phi),
                color='r',
                linestyle='--')

    return phi

def eight_figure(reactions, frac, ax, c, equal, maxi_norm=False):
    if reactions[-1] == 'entity':
        exception_msg = 'The eight is a trajectory of a moving solid, but your' \
                      + ' entry is an entity (i.e. fixed body).'
        raise Exception(exception_msg)

    times, X, Y = reactions[:3]

    Nt = len(times)

    # Same notice as in the previous function find_period.
    i0 = int(frac*Nt)

    x, y = X[i0:]/0.075, Y[i0:]
#    x, y = X[i0:], Y[i0:]

    width  = max(x) - min(x)
    height = max(y) - min(y)
    ratio  = height/width

    print('width/2 = %0.6f' % (width/2.))
    print('ratio   = %0.6f' % ratio)

    m = np.mean(x)
    
    if maxi_norm:
        maxi_x = np.max(x-m)
        maxi_y = np.max(y)
    else:
        maxi_x = 1
        maxi_y = 1

    ax.plot((x - m)/maxi_x + c, y/maxi_y, linestyle='-', color='black')

#    ax.set_xlabel(r'$X/D$', fontsize=fontsize)
    ax.set_xlabel(r'$U_{\mathrm{r}}$', fontsize=fontsize)
    ax.set_ylabel(r'$Y/D$', fontsize=fontsize)
    
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

#    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    ax.set_xticks(range(3,11))
#    ax.set_yticks([-5e-1,0,5e-1])

    if equal:
        ax.axis('equal')

    annotate = False
    if annotate:
        x1 = min(x) - m
        x2 = max(x) - m

        y1_arrow = 1.1*min(y)
        y1_text = 1.1*y1_arrow

        ax.annotate(s='', xy=(x1,y1_arrow), xytext=(x2,y1_arrow),
                    arrowprops=dict(arrowstyle='<->'))

        ax.text(x= (x1 + x2)/2.,
                y=y1_text,
                s=r'$2\bar{X}_{\mathrm{max}}$',
                color='black',
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize)

#    ax.set_ylim([-.9,.9])
#    ax.set_ylim([-.7,.7])
#    ax.set_xlim([-.09,.09])

def find_upper_shell(reactions, frac, ax, equal):
    if reactions[-1] == 'entity':
        exception_msg = 'The upper shell is the upper trajectory of a moving '\
                      + 'solid, but your entry is an entity.'
        raise Exception(exception_msg)

    times, X, Y = reactions[:3]
    U, V = reactions[4:6]

    speed = np.sqrt(U**2 + V**2)

    Nt = len(times)

    i0 = int(frac*Nt)
#    x, y = X[i0:], Y[i0:]

    imin, imax = 0, 0

    # Countercurrent at outer shell
#    for i in range(i0, Nt-1):
#        if X[i] > max(X[i-1], X[i+1]):
#            imax = i
#            break
#
#    for i in range(imax + 1, Nt-1):
#        if X[i] < min(X[i-1], X[i+1]):
#            imin = i
#            break
#
#    ax.plot(X[imax:imin+1], Y[imax:imin+1], 'oc', linewidth=2)

    # Countercurrent at inner shell
    for i in range(i0, Nt-1):
        if X[i] < min(X[i-1], X[i+1]):
            imin = i
            break

    for i in range(imin + 1, Nt-1):
        if X[i] > max(X[i-1], X[i+1]):
            imax = i
            break

    ax.plot(X[imin:imax+1], Y[imin:imax+1], color='cyan', marker='o', linewidth=2)

#    imax_speed = imin
#    for i in range(imax + 1, Nt-1):
#    for i in range(imax + 1, imin-1):
#        if speed[i] > max(speed[i-1], speed[i+1]):
#            imax_speed = i
#            break

#    ax.plot(X[imax:], Y[imax:], color='black', linewidth=0.5)
    ax.plot(X[i0:], Y[i0:], color='black', linewidth=0.5)

#    ax.plot([X[imax_speed]], [Y[imax_speed]], color='red', marker='o',
#            label='Position of maximum speed')

    ax.plot([X[i0]], [Y[i0]], color='black', marker='o',
            label='Starting point')
    ax.plot([X[i0+5]], [Y[i0+5]], color='gray', marker='o')

#    ax.legend(loc='best', fontsize=12, numpoints=1)

    if equal:
        ax.axis('equal')

    tg = abs(Y[imax]/Y[imin])
    p = (2./np.pi)*np.arctan(tg)
    print('p = %.6f' % p)

#    max_velocity = max(speed[imax:imin+1])
#    print('||Umax|| = ' + str(max_velocity))

def travel_length(Xmax, Ymax, p):
    AR = Ymax/Xmax

    zeta = np.linspace(-.999, .999, 100)

#    tmp = np.sin(p*np.pi/2.)/np.sqrt(1 + zeta) - np.cos(p*np.pi/2.)/np.sqrt(1 - zeta)
    tmp = np.sin(p*np.pi/2.)/np.sqrt(1 + zeta) + np.cos(p*np.pi/2.)/np.sqrt(1 - zeta)

    tmp *= (1/8.)*AR

    integrand = np.sqrt(1 + tmp**2)

    integral = trapz(integrand, zeta, axis=0)

    return Xmax*integral

