import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline
import os
import json

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
    Gamma_ch = k**2/(k**2+T**2+2*T*k)
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
            # Fallback por si ejecutas en otro entorno o necesitas debugear
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

def background_full(f, M, Cg,k, Mpl=1, g=106.75, c=3, m=0, N=90):
    phi_end = root_scalar(
    condicion_fin,
    args=(f, M, Mpl, g, Cg, c, m, k),
    bracket=[0.01, np.pi*f],
    method='brentq'
    ).root
    
    Q_end = Qphi(phi_end, f, M, Mpl, g, Cg, c, m, k)

    sol = solve_ivp(
        fun=dYdN,
        t_span=(0, -N),
        y0=[phi_end, Q_end],
        args=(f, M, Mpl, g, Cg, c, m),
        method='RK45',
        rtol=1e-8, atol=1e-10
    )

    phi_N = sol.y[0][::-1]   # invertir si quieres que N crezca
    Q_N   = sol.y[1][::-1]
    N_vals = N + sol.t[::-1]  #ahora va de 0 a 90

    H_N = np.array([Hubble(phi, f, M, Mpl) for phi in phi_N])
    T_N = np.array([temperatura(phi, Q, f, M, Mpl, g) for phi, Q in zip(phi_N, Q_N)])
    TH_N = T_N / H_N

    return N_vals, phi_N, Q_N, TH_N, T_N, H_N

def find_M(f, Cg,k, M_seed = None):
    def objetivo(M):
        N_vals, phi_N, Q_N, TH_N, T_N, H_N = background_full(f,M,Cg,k)
        N_pivot = 30 #total 90 efolds, queremos 60 desde escala pivot, es decir, han pasado 20 efolds
        idx_pivot = np.argmin(np.abs(N_vals - N_pivot)) #indice escala pivote
        phi_star = phi_N[idx_pivot]
        Q_star   = Q_N[idx_pivot]
        As = espectro_potencias(phi_star, Q_star, f, M, Mpl, g,c,k)
        As_obs = 2.10*10**-9
        As_obs_err = 0.03*10**-9
        return As/As_obs - 1
    if M_seed is None:
        sol = root_scalar(objetivo, bracket=[1e-9,1e-2], method='brentq')
    else:
        x0 = M_seed
        x1 = M_seed*1.1
        try:
            sol = root_scalar(objetivo,method='secant', x0=x0, x1=x1)
            if not sol.converged:
                raise RuntimeError("Secant no convergió")
        except:
            M_lower = max(1e-9, M_seed*0.5)
            M_upper = min(1e-2, M_seed*2)
            sol = root_scalar(objetivo, bracket=[M_lower, M_upper], method='brentq')

    N_vals, phi_N, Q_N, TH_N, T_N, H_N = background_full(f,sol.root,Cg,k)
    return sol.root,N_vals, phi_N, Q_N, TH_N, T_N, H_N

from tqdm import tqdm
# Diccionario anidado
# cada 'f' le corresponde un'Cgamma' y a cada 'Cgamma' le corresponde un array de 'K'

f_vals = np.array([0.8, 1, 1.5, 1.75, 2, 2.5, 3, 4, 5])
Cg_vals = np.logspace(2, 12, 100)
k_vals = np.logspace(-15, -1, 50)

parametros = {
    f: {
        Cg: k_vals
        for Cg in Cg_vals
    }
    for f in f_vals
}

Mpl = 1
g = 106.75
c = 3
m = 0
kpivot = 0.05 # Mpc^-1



evolucion_k = {}
resultados_escalares = {f: {'Cg': [], 'K':[],'ns': [], 'As': [], 'r': [], 
                            'Pt': [], 'Q_star': [],'phi_star': [], 'M': [],
                            'phi_SWIM': [], 'Q_SWIM': []
                            } for f in parametros.keys()}


total_iter = sum(
    len(parametros[f][Cg]) 
    for f in parametros 
    for Cg in parametros[f]
)


with tqdm(total=total_iter) as pbar:
    for f in parametros:
        M_seed = None  
        for Cg in parametros[f]:
            for K in parametros[f][Cg]:
                try:
      
                    M, N_vals, phi_N, Q_N, TH_N, T_N, H_N = find_M(f, Cg,K)
                    
                    M_seed = M  # Actualizamos la semilla para la siguiente iteración
                    N_pivot = 30 #total 90 efolds, queremos 60 desde escala pivot, es decir, han pasado 20 efolds
                    idx_pivot = np.argmin(np.abs(N_vals - N_pivot)) #indice escala pivote
                    phi_star = phi_N[idx_pivot]
                    Q_star   = Q_N[idx_pivot]
                    
                    phi_SWIM = phi_N[0]
                    Q_SWIM = Q_N[0]
              
                    
                    At = espectro_tensorial(phi_star, f, M, Mpl)
                    As = espectro_potencias(phi_star, Q_star, f, M, Mpl, g, c,K)
                    r = At/As
                    ns = indice_espectral(phi_star, Q_star, f, M, Mpl, g, c, m,K)
                
                    
                    
                    resultados_escalares[f]['Cg'].append(Cg)
                    resultados_escalares[f]['K'].append(K)
                    resultados_escalares[f]['Q_star'].append(Q_star)
                    resultados_escalares[f]['phi_star'].append(phi_star)
                    resultados_escalares[f]['ns'].append(ns)
                    resultados_escalares[f]['As'].append(ps)
                    resultados_escalares[f]['Pt'].append(pt)
                    resultados_escalares[f]['r'].append(r)
                    resultados_escalares[f]['M'].append(M)
                    resultados_escalares[f]['phi_SWIM'].append(phi_SWIM)
                    resultados_escalares[f]['Q_SWIM'].append(Q_SWIM)
                    
                    
                    HN = np.array([Hubble(phi, f, M, Mpl) for phi in phi_N])
                    
                    
                    k_ar = np.array([kpivot * np.exp(N-N_vals[idx_pivot]) * (H/HN[idx_pivot]) for N, H in zip(N_vals, HN)])
                    
                    Pr_array = []
                    ns_array = []
                    ev_ar = []
                    eta_ar = [] 
                    for phi, Q in zip(phi_N, Q_N):
                        
                        Pr_array.append(espectro_potencias(phi, Q, f, M, Mpl, g,c,K))
                        ns_array.append(indice_espectral(phi, Q, f, M, Mpl, g, c, m,K))
                        ev_ar.append(epsilon_v(phi, f, M, Mpl))
                        eta_ar.append(abs(eta_v(phi, f, M, Mpl)))
                        
                    Pr_array = np.array(Pr_array)
                    ns_array = np.array(ns_array)
                    
                    
                    evolucion_k[(f, Cg,K)] = {
                        'N_vals': N_vals, 'phi_N': phi_N, 'Q_N': Q_N, 'TH_N': TH_N,
                        'k_ar': k_ar, 'As': As, 'ns': ns_array, 'Pr': Pr_array, 
                        'ev': ev_ar, 'eta': eta_ar, 'Q_star': Q_star, 'ns_pivot': ns,
                        'r': r, 'M': M, 'T_N': T_N, 'H_N': H_N, 'phi_SWIM': phi_SWIM, 'Q_SWIM': Q_SWIM
                    }
                    
                    
                except (UnboundLocalError, RuntimeError, ValueError):
                    pass
            
                pbar.update(1)

evolucion_k_serializable = {
    str(k): {key: (v.tolist() if isinstance(v, np.ndarray) else v) for key,v in val.items()}
    for k,val in evolucion_k.items()
}

with open("wi_natural_mod.json", "w") as f:
    json.dump(evolucion_k_serializable, f, indent=2)
