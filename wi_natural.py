import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from tqdm import tqdm

def pot(phi, f, M):
    return M**4*(1+np.cos(phi/f))

def dpot(phi, f, M):
    return -M**4*np.sin(phi/f)/f

def ddpot(phi,f, M):
    return -M**4*np.cos(phi/f)/f**2

####    PARAMETROS SLOWROLL     ####

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

####    ECS SLOWROLL    ####

def Hubble(phi, f, M, Mpl):
    V = pot(phi, f, M)
    return np.sqrt(V/(3*Mpl**2))

def phi_dot(phi, Q,f,M,Mpl):
    dV = dpot(phi,f,M)
    H = Hubble(phi,f,M,Mpl)
    return -dV/(3*H*(1+Q))

def temperatura(phi, Q,f,M,Mpl, g):
    dphi = phi_dot(phi, Q,f,M,Mpl)
    return ((3*Q*dphi**2*30)/(4*g*np.pi**2))**(1/4)

def dist_BE(phi, Q,f,M,Mpl,g):
    T = temperatura(phi, Q,f,M,Mpl,g)
    H = Hubble(phi,f,M,Mpl)
    return 1/(np.exp(H/T)-1)

####    RELACION PHI Y Q    ####

def f_q(Q,c):
    return Q*(1+Q)**(c/4)/Q**(c/4)

def dfq(Q,c):
    return (1 + Q)**(c/4) * Q**(-c/4 - 1) * ((c - 4)*Q + c) / 4

def Aphi(phi,f,M,Mpl,g, Cg, c,m):
    H = Hubble(phi,f,M,Mpl)
    V = pot(phi,f,M)
    dV = dpot(phi,f,M)
    return Cg*(15*Mpl**2*dV**2/(2*np.pi**2*g*V))**(c/4)*phi**(m)/(3*H)

####    METODO DE NEWTON    ####

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

####    CALCULO DE Q(PHI)    ####

def Qphi(phi,f,M,Mpl,g, Cg, c,m):
    
    A = Aphi(phi,f,M,Mpl,g, Cg, c,m)
        
    Q = A**(4/(4-c))
        
    fq = f_q(Q,c)
    df = dfq(Q,c)
        
    fw = weak_fq(Q,c)
    dfw = weak_dfq(Q,c)
    
    Fq = lambda Q: f_q(Q,c)-A
    df = lambda Q: dfq(Q,c)
        
    Q = newton(Fq,df,Q, c,tol=1e-3,iter=1000)
    Qw = 0 #newton(Fq,dfq,Q, c,tol=1e-5,iter=1000)
    
    return Q, Qw 

####    FINAL DE INFLACION  #####

def fin_inflacion(phi_val , Q ,f,M,Mpl,g, Cg, c,m):
    
    for phi, q in zip(phi_val, Q):
        ev = epsilon_v(phi,f,M,Mpl)
        if abs(ev/(1+q) - 1) < 1e-2:
            #print("campo final: ", phi)
            #print("Q final: ", q)
            phi_end = phi
            q_end = q
    
    return phi_end, q_end

####    INICIO DE INFLACIÓN ####

def dYdN(N, Y, f, M, Mpl, g, Cg, c, m):
    phi, Q = Y
    H = Hubble(phi, f, M, Mpl)
    dphi = -dpot(phi,f,M) / (3*H**2 * (1 + Q))
    dlnQ = dlnQdN(phi, Q, f, M, Mpl, c, m)
    return [dphi, Q*dlnQ]

####    ECUACIONES DINÁMICAS    #####

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

####    OBSERVABLE CMB  #####

def espectro_potencias(phi, Q,f,M,Mpl,g, Gi):
    H =Hubble(phi,f,M,Mpl)
    dphi = phi_dot(phi, Q, f, M, Mpl)
    T = temperatura(phi, Q,f,M,Mpl,g)
    n = dist_BE(phi, Q,f,M,Mpl,g)
    
    return (H/dphi)**2*(H/(2*np.pi))**2*(1+2*n+(2*np.pi*Q*np.sqrt(3)*T)/(H*np.sqrt(3+4*np.pi*Q)))*Gi
def indice_espectral(phi, Q,f,M,Mpl,g, c, m, dlnGi):
    ev = epsilon_v(phi,f,M,Mpl)
    etv = eta_v(phi,f,M,Mpl)
    dlnTH = dlnTHdN(phi, Q,f,M,Mpl, c, m)
    dlnQ = dlnQdN(phi, Q,f,M,Mpl, c, m)
    n = dist_BE(phi, Q,f,M,Mpl,g)
    T = temperatura(phi,Q,f,M,Mpl,g)
    H = Hubble(phi,f,M,Mpl)
    
    B = 1+2*n + (T/H)*(2*Q*np.pi*np.sqrt(3))/(np.sqrt(3+4*np.pi*Q))
       
    A = 2*np.pi*np.sqrt(3)*Q/(np.sqrt(3+4*np.pi*Q))
    dA = A*(dlnQ - Q*dlnQ *2*np.pi/(3+4*np.pi*Q))
    
    dn = np.exp(H/T)*n**2*(H/T)*dlnTH
    
    dlnB = (2*dn + (T/H)*(dA + A*dlnTH))/B
    
    return 1 - 6*ev/(1+Q) + 2*etv/(1+Q) + 2*Q*dlnQ/(1+Q) + dlnB + dlnQ*Q*dlnGi
    

def espectro_tensorial(phi,f,M,Mpl):
    H = Hubble(phi,f,M,Mpl)
    
    return 8* (H/(2*np.pi))**2/Mpl**2

def background_full(f, M, Cg, Mpl=1, g=106.75, c=3, m=0, N=60):
    # 1. Barrido en phi para encontrar phi_end
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

    # 2. Encontrar phi_end
    phi_end, Q_end = fin_inflacion(phi_val, Q_list, f, M, Mpl, g, Cg, c, m)
    
    
    
    # 3. Integrar hacia atrás para N e-folds
    
    sol = solve_ivp(
        fun=dYdN,
        t_span=(0, -N),
        y0=[phi_end, Q_end],
        args=(f, M, Mpl, g, Cg, c, m),
        method='RK45',
        rtol=1e-8, atol=1e-10
    )

    """phi_N = sol.y[0]
    N_vals = sol.t  # t es negativo: 0 -> -N
    #asi N[0] = 0 fin y N[-1] = -60 inicio
    #lo invertimos para tener el inicio en el primer elemento
    N_vals = N_vals[::-1]"""
    phi_N = sol.y[0][::-1]   # invertir si quieres que N crezca
    Q_N   = sol.y[1][::-1]
    N_vals = sol.t[::-1] 

    # 4. Calcular Q(N) y T/H(N)
    #Q_N = np.array([Qphi(phi, f, M, Mpl, g, Cg, c, m)[0] for phi in phi_N])
    H_N = np.array([Hubble(phi, f, M, Mpl) for phi in phi_N])
    T_N = np.array([temperatura(phi, Q, f, M, Mpl, g) for phi, Q in zip(phi_N, Q_N)])
    TH_N = T_N / H_N
    
   #como hemos invertido N_vals hay que invertir todos los arrays 
    
    """Q_N = Q_N[::-1]
    TH_N = TH_N[::-1]
    phi_N = phi_N[::-1]"""

    return N_vals, phi_N, Q_N, TH_N, T_N, H_N

def find_M(f, Cg):
    def objetivo(M):
        #M = np.exp(logM)
        N_vals, phi_N, Q_N, TH_N, T_N, H_N = background_full(f,M,Cg)
        #phi_star = phi_N[-1]
        #Q_star   = Q_N[-1]
        phi_star = phi_N[0]
        Q_star   = Q_N[0]
        G3 = 1+4.981*Q_star**1.946+0.127*Q_star**4.33
        As = espectro_potencias(phi_star, Q_star, f, M, Mpl, g, G3)
        As_obs = 2.10*10**-9
        As_obs_err = 0.03*10**-9
        return As/As_obs - 1
        #return np.log(As) - np.log(As_obs)
    
    sol = root_scalar(objetivo, bracket=[1e-15,1], method='brentq')


    N_vals, phi_N, Q_N, TH_N, T_N, H_N = background_full(f,sol.root,Cg)
    return sol.root,N_vals, phi_N, Q_N, TH_N, T_N, H_N

# cada 'f' le corresponde un'Cgamma' 
parametros = {
    #1: np.logspace(3, 10, 200),
    1.5: np.logspace(3, 10, 200),
    #1.75: np.logspace(3, 10, 200),
    #2: np.logspace(3, 10, 200),
    #2.5: np.logspace(3, 10, 30), 
    #3: np.logspace(3, 10, 30),
    4: np.logspace(3, 10,10)
}

Mpl = 1
g = 106.75
c = 3
m = 0
kpivot = 0.05 # Mpc^-1

# Estructuras para almacenar los resultados
evolucion_k = {}
resultados_escalares = {f: {'Cg': [], 'ns': [], 'As': [], 'r': [], 'Pt': [], 'Q_star': []} for f in parametros.keys()}

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
                
                # --- Observables en la escala pivote ---
                G3 = 1 + 4.981*Q_star**1.946 + 0.127*Q_star**4.33
                dlnG3 = ((4.981*1.946)*Q_star**0.946 + (0.127*4.33)*Q_star**3.33)/G3
                
                pt = espectro_tensorial(phi_star, f, M, Mpl)
                ps = espectro_potencias(phi_star, Q_star, f, M, Mpl, g, G3)
                r = pt/ps
                ns = indice_espectral(phi_star, Q_star, f, M, Mpl, g, c, m, dlnG3)
                
                # Guardamos los valores escalares
                resultados_escalares[f]['Cg'].append(Cg)
                resultados_escalares[f]['Q_star'].append(Q_star)
                resultados_escalares[f]['ns'].append(ns)
                resultados_escalares[f]['As'].append(ps)
                resultados_escalares[f]['Pt'].append(pt)
                resultados_escalares[f]['r'].append(r)
                
                # --- Evolución dependiente de N ---
                HN = np.array([Hubble(phi, f, M, Mpl) for phi in phi_N])
                
                k_ar = np.array([kpivot * np.exp(N+60) * (H/HN[0]) for N, H in zip(N_vals, HN)])
                
                As_array = []
                ns_array = []
                ev_ar = []
                eta_ar = [] 

                for phi, Q in zip(phi_N, Q_N):
                    G3_q = 1 + 4.981*Q**1.946 + 0.127*Q**4.33
                    dlnG3_q = ((4.981*1.946)*Q**0.946 + (0.127*4.33)*Q**3.33)/G3_q
                    
                    As_array.append(espectro_potencias(phi, Q, f, M, Mpl, g, G3_q))
                    ns_array.append(indice_espectral(phi, Q, f, M, Mpl, g, c, m, dlnG3_q))
                    ev_ar.append(epsilon_v(phi, f, M, Mpl))
                    eta_ar.append(abs(eta_v(phi, f, M, Mpl)))
                    
                As_array = np.array(As_array)
                ns_array = np.array(ns_array)
                
                Pr_ar = As_array * (k_ar/kpivot)**(ns_array - 1)
                
                evolucion_k[(f, Cg)] = {
                    'N_vals': N_vals, 'phi_N': phi_N, 'Q_N': Q_N, 'TH_N': TH_N,
                    'k_ar': k_ar, 'As': As_array, 'ns': ns_array, 'Pr': Pr_ar, 
                    'ev': ev_ar, 'eta': eta_ar, 'Q_star': Q_star, 'ns_pivot': ns
                }
                
            except (UnboundLocalError, RuntimeError):
                pass
            
            pbar.update(1)
            
            

# --- Gráfica del valor escalar ns vs Cgamma para cada f ---
plt.figure(figsize=(8, 5))
for f, data in resultados_escalares.items():
    if len(data['Q_star']) > 0:
        # Emparejamos Q_star y ns, y los ordenamos de menor a mayor Q_star
        Q_sorted, ns_sorted = zip(*sorted(zip(data['Q_star'], data['ns'])))
        
        # Representamos Q_star en el eje X (escala logarítmica) y ns en el Y
        plt.semilogx(Q_sorted, ns_sorted, marker='o', linestyle='-', label=rf'$f = {f}$')

ns_ACT = 0.964
sigma_ACT = 0.02 #2sigmas

plt.axhline(ns_ACT, color='black', linestyle=':', alpha=0.7, label='ACT')

# Margen superior e inferior a 1 sigma (discontinuas)
plt.axhline(ns_ACT + sigma_ACT, color='gray', linestyle='--', alpha=0.8, label=r'Límites $2\sigma$')
plt.axhline(ns_ACT - sigma_ACT, color='gray', linestyle='--', alpha=0.8)
plt.xlabel(r"$Q_*$")
plt.ylabel(r"$n_s$ (escala pivote)")
plt.title(r"Índice espectral $n_s$ en función del parámetro de disipación $Q_*$")
plt.legend()
#plt.grid(True)
plt.show()

# --- Evolución de Pr con k ---
plt.figure(figsize=(8, 5))
pares_graficados = 0
for (f, Cg), data in evolucion_k.items():
    ns_val = data['ns_pivot']
    if ns_ACT - sigma_ACT <= ns_val <= ns_ACT + sigma_ACT:
        Q_s = data['Q_star']
        plt.loglog(data['k_ar'], data['As'], label=rf'$f={f}, Q_*={Q_s:.1e}$')
        pares_graficados += 1
if pares_graficados == 0:
    plt.text(0.5, 0.5, r'Ningún par $(f, Q_*)$ es compatible' + '\n' + r'con $n_s$ a $2\sigma$', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12)
else:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(1e-14, 1)
plt.xlabel("$k$")
plt.ylabel(r"$P_R$")
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.grid(True)
plt.tight_layout()
plt.show()
