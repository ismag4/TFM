import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline
import os
from tqdm import tqdm
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

def f_q(Q,c):
    return Q*(1+Q)**(c/4)/Q**(c/4)

def dfq(Q,c):
    return (1 + Q)**(c/4) * Q**(-c/4 - 1) * ((c - 4)*Q + c) / 4

def Aphi(phi,f,M,Mpl,g, Cg, c,m):
    H = Hubble(phi,f,M,Mpl)
    V = pot(phi,f,M)
    dV = dpot(phi,f,M)
    return Cg*(15*Mpl**2*dV**2/(2*np.pi**2*g*V))**(c/4)*phi**(m)/(3*H)

def newton(f, df, x0,c, tol, iter):
    
    x = x0
    for _ in range(iter):
        Fx = f(x)
        dFx = df(x)
        if dFx == 0:
            raise RuntimeError("Derivada nula")
        xn = x - Fx/dFx
        if abs(xn - x) < tol:
            return xn
        x = xn
        
    raise RuntimeError("No se alcanzó convergencia")

def Qphi(phi,f,M,Mpl,g, Cg, c,m):
    
    A = Aphi(phi,f,M,Mpl,g, Cg, c,m)
        
    Q = A**(4/(4-c))
        
    fq = f_q(Q,c)
    df = dfq(Q,c)
    
    Fq = lambda Q: f_q(Q,c)-A
    df = lambda Q: dfq(Q,c)
        
    Q = newton(Fq,df,Q, c,tol=1e-3,iter=1000)
    
    return Q

def dYdN(N, Y, f, M, Mpl, g, Cg, c, m):
    phi, Q = Y
    H = Hubble(phi, f, M, Mpl)
    dphi = -dpot(phi,f,M) / (3*H**2 * (1 + Q))
    dlnQ = dlnQdN(phi, Q, f, M, Mpl, c, m)
    return [dphi, Q*dlnQ]

def dlnQdN(phi, Q,f,M,Mpl, c ,m):
    ev = epsilon_v(phi,f,M,Mpl)
    etv = eta_v(phi,f,M,Mpl)
    kv = kappa_v(phi,f,M,Mpl)
    Cq = 4-c+(4+c)*Q
    return ((2*c+4)*ev-2*c*etv-4*m*kv)/Cq

def dlnTHdN(phi, Q,f,M,Mpl, c ,m):
    ev = epsilon_v(phi,f,M,Mpl)
    etv = eta_v(phi,f,M,Mpl)
    kv = kappa_v(phi,f,M,Mpl)
    Cq = 4-c+(4+c)*Q
    return ((7-c+(5+c)*Q)*ev/(1+Q)-2*etv-m*kv*(1-Q)/(1+Q))/Cq

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

def espectro_potencias(phi, Q,f,M,Mpl,g,c):
    H =Hubble(phi,f,M,Mpl)
    dphi = phi_dot(phi, Q, f, M, Mpl)
    T = temperatura(phi, Q,f,M,Mpl,g)
    n = dist_BE(phi, Q,f,M,Mpl,g)
    Gi, dGi = growing_mode(Q)
    
    return (H/dphi)**2*(H/(2*np.pi))**2*(1+2*n+(2*np.pi*Q*np.sqrt(3)*T)/(H*np.sqrt(3+4*np.pi*Q)))*Gi
    
def indice_espectral(phi, Q,f,M,Mpl,g, c, m):
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

def fin_inflacion(phi_val , Q ,f,M,Mpl,g, Cg, c,m):

    for phi, q in zip(phi_val, Q):
        ev = epsilon_v(phi,f,M,Mpl)
        if abs(ev/(1+q) - 1) < 1e-2:
            phi_end = phi
            q_end = q

def background_full(f, M, Cg, Mpl=1, g=106.75, c=3, m=0, N=60):
    phi_val = np.linspace(0.01, np.pi*f, 100000)

    Q_list = []
    for phi in phi_val:
        try:
            Qphi_val = Qphi(phi,f,M,Mpl, g, Cg, c,m)[0]
            if epsilon_v(phi,f,M,Mpl)/(1 + Qphi_val) <= 1:
                Q_list.append(Qphi_val)
            else:
                Q_list.append(np.nan)
        except RuntimeError:
            Q_list.append(np.nan)
    Q_list = np.array(Q_list)
    
    phi_end, Q_end = fin_inflacion(phi_val, Q_list, f, M, Mpl, g, Cg, c, m)
    
    sol = solve_ivp(
        fun=dYdN,
        t_span=(0, -N),
        y0=[phi_end, Q_end],
        args=(f, M, Mpl, g, Cg, c, m),
        method='RK45',
        rtol=1e-8, atol=1e-10
    )

    phi_N = sol.y[0][::-1] 
    Q_N   = sol.y[1][::-1]
    N_vals = sol.t[::-1]

    H_N = np.array([Hubble(phi, f, M, Mpl) for phi in phi_N])
    T_N = np.array([temperatura(phi, Q, f, M, Mpl, g) for phi, Q in zip(phi_N, Q_N)])
    TH_N = T_N / H_N

    return N_vals, phi_N, Q_N, TH_N, T_N, H_N

def find_M(f, Cg):
    def objetivo(M):
        N_vals, phi_N, Q_N, TH_N, T_N, H_N = background_full(f,M,Cg)
        phi_star = phi_N[0]
        Q_star   = Q_N[0]
        G3 = 1+4.981*Q_star**1.946+0.127*Q_star**4.33
        As = espectro_potencias(phi_star, Q_star, f, M, Mpl, g,c)
        As_obs = 2.10*10**-9
        As_obs_err = 0.03*10**-9
        return As/As_obs - 1
        
    sol = root_scalar(objetivo, bracket=[1e-15,1], method='brentq')

    N_vals, phi_N, Q_N, TH_N, T_N, H_N = background_full(f,sol.root,Cg)
    return sol.root,N_vals, phi_N, Q_N, TH_N, T_N, H_N

# cada 'f' le corresponde un'Cgamma' 
parametros = {
    #1: np.logspace(5, 17, 500),
    #1.5: np.logspace(5, 17, 500),
    #1.75: np.logspace(3, 10, 250),
    #2: np.logspace(5, 12, 1000),
    #2.5: np.logspace(3, 10, 250),
    #3: np.logspace(3, 10, 300),
    #4: np.logspace(3, 10, 400),
    5: np.logspace(3, 10, 400),
    #0.8: np.logspace(6, 20,1000)
}

Mpl = 1
g = 106.75
c = 3
m = 0
kpivot = 0.05 # Mpc^-1

# Estructuras para almacenar los resultados
evolucion_k = {}
resultados_escalares = {f: {'Cg': [], 'M':[], 'ns': [], 'As': [], 'r': [], 'Pt': [], 'phi_star':[], 'Q_star': []} for f in parametros.keys()}

# Calculamos el total de iteraciones para la barra de progreso
total_iter = sum(len(parametros[f]) for f in parametros)

# =============================================================================
# 3. BUCLE PRINCIPAL DE CÁLCULO
# =============================================================================
with tqdm(total=total_iter) as pbar:
    for f in parametros:
        for Cg in parametros[f]:
            try:
                # --- Background ---
                
                M,N_vals, phi_N, Q_N, TH_N, T_N, H_N = find_M(f, Cg)
                
                phi_star = phi_N[0]
                Q_star   = Q_N[0]
                
                At = espectro_tensorial(phi_star, f, M, Mpl)
                As = espectro_potencias(phi_star, Q_star, f, M, Mpl, g,c)
                r = At/As
                ns = indice_espectral(phi_star, Q_star, f, M, Mpl, g, c, m)
                
                resultados_escalares[f]['Cg'].append(Cg)
                resultados_escalares[f]['Q_star'].append(Q_star)
                resultados_escalares[f]['ns'].append(ns)
                resultados_escalares[f]['As'].append(As)
                resultados_escalares[f]['Pt'].append(At)
                resultados_escalares[f]['r'].append(r)
                resultados_escalares[f]['M'].append(M)
                resultados_escalares[f]['phi_star'].append(phi_star)
                
                # --- Evolución dependiente de N ---
                HN = np.array([Hubble(phi, f, M, Mpl) for phi in phi_N])
                
                k_ar = np.array([kpivot * np.exp(N+60) * (H/HN[0]) for N, H in zip(N_vals, HN)])
                
                Pr_array = []
                ns_array = []
                ev_ar = []
                eta_ar = [] 

                for phi, Q in zip(phi_N, Q_N):
                    
                    Pr_array.append(espectro_potencias(phi, Q, f, M, Mpl, g, c))
                    ns_array.append(indice_espectral(phi, Q, f, M, Mpl, g, c, m))
                    ev_ar.append(epsilon_v(phi, f, M, Mpl))
                    eta_ar.append(abs(eta_v(phi, f, M, Mpl)))
                    
                Pr_array = np.array(As_array)
                ns_array = np.array(ns_array)
                
                evolucion_k[(f, Cg)] = {
                    'N_vals': N_vals, 'phi_N': phi_N, 'Q_N': Q_N, 'TH_N': TH_N,
                    'k_ar': k_ar, 'As':As, 'ns': ns_array, 'Pr': Pr_array, 
                    'ev': ev_ar, 'eta': eta_ar, 'Q_star': Q_star, 'ns_pivot': ns
                }
                
            except (UnboundLocalError, RuntimeError):
                pass
            
            pbar.update(1)

evolucion_k_serializable = {
    str(k): {key: (v.tolist() if isinstance(v, np.ndarray) else v) for key,v in val.items()}
    for k,val in evolucion_k.items()
}

with open("wi_natural_mod_PRmaxfine_correct.json", "w") as f:
    json.dump(evolucion_k_serializable, f, indent=2)

with open("wi_natural_mod_PRmaxfine_correct.json", "w") as f:
    json.dump(evolucion_k_serializable, f, indent=2)
