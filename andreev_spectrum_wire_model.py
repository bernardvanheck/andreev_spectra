"""
Zeeman and spin-orbit effects in the Andreev spectra of nanowire junctions
Accompanying numerical code
(c) 2016-2017 Bernard van Heck, Jukka Vayrynen, Leonid Glazman (Yale).
"""

import numpy as np
from scipy.optimize import brentq,
from scipy.integrate import quad
from scipy.linalg import expm
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt


class SimpleNamespace(object):
    """A simple container for parameters.
    
    The SimpleNamespace instances required for this project
    should contain the following parameters:
    
        * regime: 'high_mu' or 'high_so'
        * a: float, spin-orbit coupling strength times kF
        * b: float, Zeeman energy
        * c: float, chemical potential
        * phi: float, phase difference across the junction
        * tau: float, transparency of the junction
        
    All energies must be expressed in units of the superconducting gap Delta_0.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0 , -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]], complex)
s0 = np.array([[1, 0], [0, 1]], complex)

# 8x8 Identity matrix
eye = np.eye(8, dtype=complex)

# Kronecker product
def kron(a, b, c):
    return np.kron(a, np.kron(b, c))

# Complex inner product
def inner(v1, v2):
    return np.dot(np.conj(v1), v2)

# Applies the particle-hole symmetry operator of the linearized model
# to an 8-dimensional complex vector.
def phs(vec):
    return kron(sy, sx, sy).dot(np.conj(vec))

def compute_gap(p):
    """
    Computes the gap of the homegenous nanowire as a function
    of system parameters.
    """
    if p.regime == 'high_mu':
        condition = (np.sqrt(p.b * (1 - p.b)) < p.a) if p.b < 1 else True
        gap1 = p.a / np.sqrt(p.a**2 + p.b **2)
        gap2 = np.sqrt(p.a**2 + (1 - p.b)**2)
        return gap1 if condition else gap2
    elif p.regime == 'high_so':
        b0 = p.c**2 / np.sqrt(1 + p.c**2)
        gap = min(np.abs(np.sqrt(1 + p.c**2) - p.b), 1) 
        return min(np.sqrt(1 - p.b**2/p.c**2), 1) if (p.b < b0) else gap 
    else:
        print("Unknown regime")
        return

def poles(en, p):
    """
    Return the two poles of the Green's function with negative
    imaginary part.
    """
    if p.regime == 'high_mu':
        z1 = -1 + en**2 + p.a**2 + p.b**2 + 2 * 1j * np.sqrt(p.a**2 * (1 - en**2) - en**2 * p.b**2 + 0j)
        z2 = -1 + en**2 + p.a**2 + p.b**2 - 2 * 1j * np.sqrt(p.a**2 * (1 - en**2) - en**2 * p.b**2 + 0j)
    elif p.regime == 'high_so':
        z1 = - 1 - p.b**2 + p.c**2 + en**2 + 2 * np.sqrt(p.b**2 + p.c**2 * (en**2 - 1) + 0j)
        z2 = - 1 - p.b**2 + p.c**2 + en**2 - 2 * np.sqrt(p.b**2 + p.c**2 * (en**2 - 1) + 0j)
    else:
        print("Unknown regime.")
        return None
    poles =[np.sqrt(z1), -np.sqrt(z1), np.sqrt(z2), -np.sqrt(z2)]
    poles_neg_im = [pl for pl in poles if pl.imag < 0]
    if len(poles_neg_im) == 2:
            return poles_neg_im
    else:
        print("No negative imaginary part!:", poles)
        print(en, compute_gap(p), p.c, p.b)
        return (np.nan, np.nan)


def g_matrix(en, p, x=0, s=-1):
    """
    Returns the real space Green's function of the wire.
    en is the energy; x is the distance from the origin;
    s indicates whether the point is on the left or the right of the junction.
    """
    q0, q1 = poles(en, p)
    if p.regime == 'high_mu':
        A0 = (- en * (1 + p.a**2 + p.b**2 - en**2) * eye
              - p.a * (1 + p.a**2 + p.b**2 - en**2) * kron(sz, sz, sz)
              + p.b * (-1 + p.a**2 + p.b**2 - en**2) * kron(s0, s0, sx)
              + (1 + p.a**2 - en**2 - p.b**2) * kron(sx, s0, s0)
              + 2 * p.b * en * kron(sx, s0, sx)
              - 2 * p.b * p.a * kron(sy, sz, sy))
        A1 = ((-1 + p.a**2 + p.b**2 + en**2) * kron(sz, sz, s0)
              + 2 * en * p.a * kron(s0, s0, sz)
              - 2 * en * p.b * kron(sz, sz, sx)
              - 2 * p.a * kron(sx, s0, sz))
        A2 = (p.a * kron(sz, sz, sz) - p.b * kron(s0, s0, sx)
              + kron(sx, s0, s0) - en * eye)
        
        g_mat = (0.5 / (q0**2 - q1**2) * 
                 (np.exp(-1j * q0 * x) * (A0 - s * A1 * q0 + A2 * q0**2 + s * kron(sz, sz, s0) * q0**3) / q0
                  - np.exp(-1j * q1 * x) * (A0 - s * A1 * q1 + A2 * q1**2 + s * kron(sz, sz, s0) * q1**3) / q1))
        return np.dot(g_mat, kron(sz, sz, s0))
    elif p.regime == 'high_so':
        B0 = (en * (en**2 - 1 - p.b**2 - p.c**2) * eye
              + (1 - en**2 - p.b**2 + p.c**2) * kron(sx, s0, s0)
              + p.c * (1 - en**2 - p.b**2 + p.c**2) * kron(sz, s0, s0)
              - p.b * (1 - p.b**2 + p.c**2 + en**2) * kron(s0, sx, sx)
              + 2 * p.b * en * kron(sx, sx, sx)
              + 2 * p.b * p.c * en * kron(sz, sx, sx))
        B1 = ((en**2 + p.c**2 - p.b**2 - 1) * kron(sz, sz, s0)
              - 2 * p.b * kron(sy, sx, sy)
              + 2 * p.c * kron(sx, sz, s0)
              - 2 * p.c * en * kron(s0, sz, s0))
        B2 = - en * eye + p.b * kron(s0, sx, sx) + kron(sx, s0, s0) - p.c * kron(sz, s0, s0)
        w = np.sqrt(1 - en**2)
        gm = (0.5 * 1j * np.exp(- w * x) / w *
              (kron(sx, s0, s0) - en * eye - 1j * s * w * kron(sz, sz, s0)))
        gm = gm.dot(expm(1j * p.c * s * x * kron(s0, sz, s0)))
        
        gp = (0.5 / (q0**2 - q1**2) *
              (np.exp(-1j * q0 * x) * (B0 - s * B1 * q0 + B2 * q0**2 + s * kron(sz, sz, s0) * q0**3) / q0
               - np.exp(-1j * q1 * x) * (B0 - s * B1 * q1 + B2 * q1**2 + s * kron(sz, sz, s0) * q1**3) / q1))
        
        Pm = 0.5 * (eye - kron(s0, sz, sz))
        Pp = 0.5 * (eye + kron(s0, sz, sz))
        g_mat = Pp.dot(gp) + Pm.dot(gm)
        return np.dot(g_mat, kron(sz, sz, s0))
    else:
        print("Unknown regime.")
        return None


def transfer_matrix(p):
    """
    Returns the transfer matrix of the junction.
    """
    r = np.sqrt(1 - p.tau) / np.sqrt(p.tau)
    t_mat = eye - 1j * r * kron(s0, sz, s0) + r * kron(s0, sy, s0)
    return t_mat


def tphi(p):
    phiexp = expm(-1j * 0.5 * p.phi *kron(sz, s0, s0))
    return np.dot(phiexp, transfer_matrix(p))


def bound_state_matrix(en, p):
    g = g_matrix(en, p)
    m = tphi(p) - eye
    return np.dot(g, m)
    

def bound_state_determinant(en, p, tol=1e-8, warn=False):
    mdet = np.linalg.det(eye - bound_state_matrix(en, p))
    if warn and (mdet.imag > tol):
        print(mdet.imag, mdet.real, mdet.imag/mdet.real)
        print(Warning("The imaginary part of the determinant "
                      + "is greater than %s" % tol))
    return mdet.real


def find_abs_energies(p, nsamples=200):
    """
    Returns the Andreev spectrum given a set of system parameters.
    
    The Andreev energies are found from the roots of the determinant equation (25).
    Roots are searched in the interval (0, gap) where gap is the excitation gap of the
    homogeneous wire given the same system parameters.
    
    The function samples the determinant function, looking for sign changing intervals.
    Roots are then determined by the brent function from SciPy.
    
    Parameters:
    -----------
    p : SimpleNamespace containing all system parameters
    nsamples : int
        Initial number of point sampled from the determinant function,
        uniformly in the interval (0, gap).
        
    Returns:
    --------
    A tuple (possibly empty) containing Andreev levels.
    """
    # Take care of zero magnetic field case
    if p.b == 0:
            Ea = np.sqrt(1 - p.tau * np.sin(p.phi/2)**2)
            return (Ea, Ea) 
    
    # Find sign-changing intervals of the determinant function
    gap = compute_gap(p)
    ens = np.linspace(1e-8, gap, nsamples, endpoint=False)
    sample = np.array([bound_state_determinant(en, p) for en in ens]) #func(xs, p)
    sign_changes = np.nonzero(np.diff(np.sign(sample)))[0]
    intervals = [[ens[s - 1], ens[s + 1]] for s in sign_changes]
    Ns = len(sign_changes)
    
    # Discard sign changes in consecutive values of the sample
    if Ns == 2 and np.diff(sign_changes)[0] == 1:
        Ns = 0
    
    # Adjust the boundary conditions when solutions
    # are very close to x=0
    if Ns:
        for s in sign_changes:
            if s == 0:
                intervals[0][0] = 0.
    if Ns == 0:
        return ()
    elif Ns == 1:
        x_1, x_2 = intervals[0]
        sol1 = brentq(lambda x: bound_state_determinant(x, p), x_1, x_2, full_output=False)
        return (sol1, )
    elif Ns == 2:
        x_1, x_2 = intervals[0]
        x_3, x_4 = intervals[1]
        sol1 = brentq(lambda x: bound_state_determinant(x, p), x_1, x_2, full_output=False)
        sol2 = brentq(lambda x: bound_state_determinant(x, p), x_3, x_4, full_output=False)
        return (sol1, sol2)
    else:
        print("# sign changes:", Ns, sign_changes)
        print(p.c, p.b)
        return ()

### Wave functions

def norm_wave_function(en, p, psi0, nsamples=200):
    """
    Computes the norm of the Andreev bound state wave function.
    """
    def psi_sq(x, s):
        psi = np.dot(np.dot(g_matrix(en, p, x, s), tphi(p) - eye), psi0)
        return np.real(inner(psi, psi))
    int1 = quad(psi_sq, 0, np.inf, args=(1))
    int2 = quad(psi_sq, 0, np.inf, args=(-1))
    return int1[0] + int2[0]
    

def wave_functions(ens, p, x=0, s=-1, nsamples=200): 
    """Returns Andreev bound state wave functions evaluated at a distance x 
    away from the junction, using Eq. (22) of the main text.
    
    Parameters:
    -----------
    ens : tuple of Andreev energies
    p : SimpleNamespace containing all system parameters
    x : float, distance from the junction, in units of zero field coherence length.
    s : either 1 or -1, for left or right side of the junction
    nsamples : int, see instructions for abs_energies()
    """
    psis = np.zeros((8, len(ens)), dtype=complex)
    for (i, en) in enumerate(ens):
        m = bound_state_matrix(en, p)
        val, vec = sla.eigs(m, k=1, sigma=1)
        vec *= np.exp(-1j * np.angle(vec[0,0]))
        norm = norm_wave_function(en, p, vec[:,0])
        psis[:,i] = np.dot(np.dot(g_matrix(en, p, x, s), tphi(p) - eye), vec[:,0])
        psis[:,i] /= np.sqrt(norm)
    return psis

### Current matrix elements

def current_matrix_elements(ens, p, nsamples=200):
    """Computes all current matrix elements involving ABS
    for a given set of system parameters.
    
    Parameters:
    -----------
    ens : tuple of Andreev energies
    p : SimpleNamespace containing all system parameters
    nsamples : int, see instructions for abs_energies()
    """
    if p.b == 0:
        EA = np.sqrt(1 - p.tau * np.sin(p.phi/2)**2)
        j11 = - 0.5 * p.tau * np.sin(p.phi) / EA
        j12p = p.tau * np.sqrt(1 - p.tau) * np.sin(p.phi/2)**2 / EA
        return (j11, j11, 0., j12p)
    psi = wave_functions(ens, p, 0, -1, nsamples)
    jop = kron(s0, sz, s0)
    if len(psi.T) == 2:
        psi1, psi2 = psi.T
        j11 = inner(psi1, jop.dot(psi1))
        j22 = inner(psi2, jop.dot(psi2))
        j12 = inner(psi1, jop.dot(psi2))
        j12p = inner(psi1, jop.dot(phs(psi2)))
        return (j11,
                j22,
                j12, j12p)
    elif len(psi.T) == 1:
        j11 = inner(psi[:,0], jop.dot(psi[:,0]))
        return (j11, np.nan, np.nan, np.nan)
    else:
        return(np.nan, np.nan, np.nan, np.nan)
    
def equilibrium_current(ens, j11, j22, fp, T):
    """
    Computes the equilibrium current of the junction.
    
    Parameters:
    -----------
    ens : Andreev energies
    fp : 1 if fermion parity is even, -1 if odd
    j11, j22 : diagonal current matrix elements
    T : float, temperature (in units of Delta_0)
    """
    if T == 0:
        j = - 0.5 * np.real(j11) - 0.5 * np.real(j22)
    else:
        f = lambda x: 1 / (1 + np.exp(x / T)) if x is not np.nan else np.nan
        n1, n2 = f(fp * ens[0]), f(ens[1])
        j = (n1 - 0.5) * fp * np.real(j11) + (n2 - 0.5) * np.real(j22)
    return j

### Convenience function for computing
def one_parameter_scan(p, var_name, vmin, vmax, N=100, plot=False, nsamples=200):
    """
    This function can be used as a shorthand to conveniently generate datasets.
    It returns energies and current matrix elements as a function of a given system parameter,
    keeping all other system parameters fixed.
    
    See also the jupyter notebook accompanying the submission for examples of usage.
    
    Parameters:
    -----------
    p : SimpleNamespace containing all system parametrs.
    var_name : string, name of system parameters to be varied
    vmin, vmax : float, initial and final value of the variable parameter
    N : int, number of datapoints
    plot : boolean, makes a quick shot of results if True
    nsamples : int, as in abs_energies()
    
    Returns:
    --------
    energies: NumPy array
        Array of ABS energies as a function of var. Missing values replaced by np.nan
    currents: NumPy array
        Array of current matrix elements as a function of var.
    """
    variables = np.linspace(vmin, vmax, N)
    energies = []
    currents = []
    gaps = []
    for var in variables:
        vars(p)[var_name] = var
        gaps.append(compute_gap(p))
        energies.append(find_abs_energies(p, nsamples))
        currents.append(current_matrix_elements(energies[-1], p, nsamples))
    maxlen = max(len(i) for i in energies) 
    energies =  np.array([i + (np.nan, ) * (maxlen - len(i)) for i in energies]).T
    currents = np.array(currents).T
    if plot:
        f, axarr = plt.subplots(1, 4, figsize=(16,4))
        #plt.figsize=(15,4)
        for ens in energies:
            axarr[0].plot(variables, gaps, color='k', linewidth=2)
            axarr[0].plot(variables, ens, linestyle='--', marker='o', markersize=3.5)
            axarr[0].set_ylim(0, 1)
        for js in currents[:2]:
            axarr[1].plot(variables, np.real(js), linestyle='--', marker='o', markersize=3.5,)
        axarr[2].plot(variables, np.abs(currents[2]), linestyle='--', marker='o', markersize=3.5, c='k')
        axarr[3].plot(variables, np.abs(currents[3]), linestyle='--', marker='o', markersize=3.5,c='k')
        for ax in axarr:
            ax.set_xlim(vmin, vmax)
            ax.set_xticks([vmin, vmin + 0.5*(vmax-vmin), vmax])
            ax.tick_params(
                axis='y',
                which='both', 
                right='off',
                left='off',
                labelright='off',
                labelleft='off')
        plt.show()
    return energies, currents