import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline
import os

def pot(phi, f, M):
    return M**4*(1+np.cos(phi/f))

def dpot(phi, f, M):
    return -M**4*np.sin(phi/f)/f

def ddpot(phi,f, M):
    return -M**4*np.cos(phi/f)/f**2

def epsilon_v(phi, f, M, Mpl):
    V = pot(phi, f, M)
    dV = dpot(phi, f, M)
    return Mpl**2*0.5*(dV/V)**2

def eta_v(phi, f, M, Mpl):
    V = pot(phi, f, M)
    ddV = ddpot(phi, f, M)
    return Mpl**2*ddV/V

def kappa_v(phi, f, M, Mpl):
    V = pot(phi, f, M)
    dV = dpot(phi, f , M)
    return Mpl**2*dV/(phi*V)

def Hubble(phi, f, M, Mpl):
    V = pot(phi, f, M)
    return np.sqrt(V/(3*Mpl**2))

def phi_dot(phi, Q,f,M,Mpl):
    dV = dpot(phi,f,M)
    H = Hubble(phi,f,M,Mpl)
    return -dV/(3*H*(1+Q))

def temperatura(phi, Q,f,M,Mpl, g):
    dphi = phi_dot(phi, Q,f,M,Mpl)
    return (3*30*Q*dphi**2/(4*g*np.pi**2))**(0.25)

def dist_BE(phi, Q,f,M,Mpl,g):
    T = temperatura(phi, Q,f,M,Mpl,g)
    H = Hubble(phi,f,M,Mpl)
    return 1/(np.expm1(H/T))

def Gamma(phi, Q,f,M,Mpl, g, Cg, c,m,k):
    T = temperatura(phi, Q,f,M,Mpl, g)
    Gamma_sph = Cg * T**c*phi**m
    Gamma_ch = k**2/(k**2+T**2)
    return Gamma_sph*Gamma_ch

def Qphi(phi,f,M,Mpl, g, Cg, c,m,k):
    def FQ(Q):
        H = Hubble(phi,f,M,Mpl)
        Gamma_total = Gamma(phi, Q,f,M,Mpl, g, Cg, c,m,k)
        return Q-Gamma_total/(3*H)
    sol = root_scalar(FQ, bracket=[1e-50, 1e3])
    return sol.root

def dlnQdN(phi, Q,f,M,Mpl, c ,m):
    ev = epsilon_v(phi,f,M,Mpl)
    etv = eta_v(phi,f,M,Mpl)
    kv = kappa_v(phi,f,M,Mpl)
    Cq = 4.0-c+(4.0+c)*Q
    return ((2*c+4)*ev-2*c*etv-4*m*kv)/Cq

def dlnTHdN(phi, Q,f,M,Mpl, c ,m):
    ev = epsilon_v(phi,f,M,Mpl)
    etv = eta_v(phi,f,M,Mpl)
    kv = kappa_v(phi,f,M,Mpl)
    Cq = 4-c+(4+c)*Q
    return (((7-c+(5+c)*Q)*ev)/(1+Q)-2*etv-m*kv*(1-Q)/(1+Q))/Cq

def dYdN(N, Y, f, M, Mpl, g, Cg, c, m):
    phi, Q = Y
    H = Hubble(phi, f, M, Mpl)
    dphi = -dpot(phi,f,M) / (3*H**2 * (1 + Q))
    dlnQ = dlnQdN(phi, Q, f, M, Mpl, c, m)
    return [dphi, Q*dlnQ]

_GQ_SWIM_FUNC = None

def get_growing_mode_spline():
    global _GQ_SWIM_FUNC
    if _GQ_SWIM_FUNC is None:
        path_datos = "GQ_smooth.dat"
        if not os.path.exists(path_datos):
            raise FileNotFoundError(f"No se encontró el archivo de modos de crecimiento en: {path_datos}")

        Q_swim, GQ_swim = np.loadtxt(path_datos, delimiter=',', unpack=True)
        _GQ_SWIM_FUNC = CubicSpline(Q_swim, GQ_swim, extrapolate=True)
    return _GQ_SWIM_FUNC

def growing_mode(Q):
    GQ_swim_func = get_growing_mode_spline()
    dG_dQ = GQ_swim_func.derivative(nu=1)
    GQ = GQ_swim_func(Q)
    dlnG = dG_dQ(Q)/GQ
    return GQ, dlnG

def espectro_potencias(phi, Q,f,M,Mpl,g,c,k):
    H =Hubble(phi,f,M,Mpl)
    dphi = phi_dot(phi, Q, f, M, Mpl)
    T = temperatura(phi, Q,f,M,Mpl,g)
    n = dist_BE(phi, Q,f,M,Mpl,g)
    Gi, dGi = growing_mode(Q)

    return (H/dphi)**2*(H/(2*np.pi))**2*(1+2*n+(2*np.pi*Q*np.sqrt(3)*T)/(H*np.sqrt(3+4*np.pi*Q)))*Gi

def indice_espectral(phi, Q,f,M,Mpl,g, c, m, k):
    ev = epsilon_v(phi,f,M,Mpl)
    etv = eta_v(phi,f,M,Mpl)
    dlnTH = dlnTHdN(phi, Q,f,M,Mpl, c, m)
    dlnQ = dlnQdN(phi, Q,f,M,Mpl, c, m)
    n = dist_BE(phi, Q,f,M,Mpl,g)
    T = temperatura(phi,Q,f,M,Mpl,g)
    H = Hubble(phi,f,M,Mpl)
    Gi, dlnGi = growing_mode(Q)

    B = 1+2*n + (T/H)*(2*Q*np.pi*np.sqrt(3))/(np.sqrt(3+4*np.pi*Q))

    A = 2*np.pi*np.sqrt(3)*Q/(np.sqrt(3+4*np.pi*Q))
    dlnA = dlnQ * (1 - (2*np.pi*Q)/(3+4*np.pi*Q))

    dn = n*(n+1)*(H/T)*dlnTH

    dlnB = (2*dn + A*(T/H)*(dlnA + dlnTH))/B

    return 1 - 6*ev/(1+Q) + 2*etv/(1+Q) + 2*Q*dlnQ/(1+Q) + dlnB + dlnQ*Q*dlnGi


def espectro_tensorial(phi,f,M,Mpl):
    H = Hubble(phi,f,M,Mpl)

    return 8* (H/(2*np.pi))**2/Mpl**2

def condicion_fin(phi, f, M, Mpl, g, Cg, c, m, k):
    Q = Qphi(phi, f, M, Mpl, g, Cg, c, m, k)
    return epsilon_v(phi, f, M, Mpl)/(1+Q) - 1

def loglike(_self, f, logCg, logk, M):

    Cg = 10**logCg
    k = 10**logk

    Mpl = 1
    c = 3
    m = 0
    N = 60
    g = 106.75

    try:

        phi_end = root_scalar(
            condicion_fin,
            args=(f,M,Mpl,g,Cg,c,m,k),
            bracket=[1e-6,np.pi*f-1e-6]
        ).root

        Q_end = Qphi(
            phi_end,f,M,Mpl,
            g,Cg,c,m,k
        )

        sol = solve_ivp(
            dYdN,
            [0,-N],
            [phi_end,Q_end],
            args=(f,M,Mpl,g,Cg,c,m),
            dense_output=True
        )

        phi_star,Q_star = sol.sol(-60)

        As = espectro_potencias(
            phi_star,Q_star,
            f,M,Mpl,g,c,k
        )

        ns = indice_espectral(
            phi_star,Q_star,
            f,M,Mpl,g,c,m,k
        )

        PT = espectro_tensorial(
            phi_star,f,M,Mpl
        )

        r = PT/As

        if not np.isfinite(ns) or not np.isfinite(As) or not np.isfinite(r):
            return -np.inf

        chi2_ns = (
        (ns-0.9626)/0.0057
        )**2

        chi2_As = (
            (np.log(1e10*As)-3.044)/0.014
        )**2

        if r < 0:
            return -np.inf
        elif r > 0.014:
             chi2_r = ((r-0.014)/0.011)**2
        else: 
            chi2_r = ((r-0.014)/0.010)**2
        
        
         chi2 = chi2_ns + chi2_As + chi_r

#        print(f'ns={ns} , As = {As}, r={r}')

        return -0.5*chi2

    except Exception as e:
        #print(e)
        return -np.inf
