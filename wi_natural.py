import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

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

def dphi_dN(N, y, f, M, Mpl, g, Cg, c, m):
    phi = y[0]   # valor escalar de phi en este paso
    Q, Qw = Qphi(phi, f, M, Mpl, g, Cg, c, m)  # calcular Q
    V_phi = dpot(phi, f, M)
    H = Hubble(phi, f, M, Mpl)
    
    return [- V_phi / (3 * H**2 * (1 + Q))]####    INICIO DE INFLACIÓN ####

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
    #G3 = 1+4.981*Q**1.946+0.127*Q**4.33

    return (H/dphi)**2*(H/(2*np.pi))**2*(1+2*n+(2*np.pi*Q*np.sqrt(3)*T)/(H*np.sqrt(3+4*np.pi*Q)))*Gi

def indice_espectral(phi, Q,f,M,Mpl,g, c, m, dlnGi):
    ev = epsilon_v(phi,f,M,Mpl)
    etv = eta_v(phi,f,M,Mpl)
    dlnTH = dlnTHdN(phi, Q,f,M,Mpl, c, m)
    dlnQ = dlnQdN(phi, Q,f,M,Mpl, c, m)
    n = dist_BE(phi, Q,f,M,Mpl,g)
    T = temperatura(phi,Q,f,M,Mpl,g)
    H = Hubble(phi,f,M,Mpl)

    B = 1+2*n + (2*Q*T*np.pi*np.sqrt(3))/(H*np.sqrt(3+4*np.pi*Q))
    A = 2*np.pi*np.sqrt(3)*Q/(np.sqrt(3+4*np.pi*Q))
    #dlnA = 1/Q - 2*np.pi/(3+4*np.pi*Q)
    dA = A*(1-2*Q*np.pi/(3+4*np.pi*Q))*dlnQ 
    #dn = n*dlnTH*(H/T)/(1-np.exp(-H/T))
    dn = n*(n+1)*dlnTH*(H/T)
    #corch = 2*dn + (T/H)*A*Q*dlnQ*dlnA+A*(T/H)*dlnTH
    corch = 2*dn + dA*T/H + A*T/H*dlnTH
    
    print("Cold: ", 1-6*ev+2*etv)
    print("sum 1: ", Q*dlnQ/(1+Q))
    print("sum 2: ", Q*dlnQ*dlnGi)
    print("sum 3: ", corch/B)
    print("T/H: ", T/H)
    print("Warm: ", 1-6*ev+2*etv + Q*dlnQ/(1+Q) + Q*dlnQ*dlnGi + corch/B)
    
    return 1-6*ev+2*etv + Q*dlnQ/(1+Q) + Q*dlnQ*dlnGi + corch/B
    

def espectro_tensorial(phi,f,M,Mpl):
    H = Hubble(phi,f,M,Mpl)
    
    return 8* (H/(2*np.pi))**2/Mpl**2

def background_full(f, M, Cg, Mpl=1, g=100, c=3, m=0, N=60):
    # 1. Barrido en phi para encontrar phi_end
    phi_val = np.linspace(0, np.pi*f, 300000)

    Q_list = []
    for phi in phi_val:
        try:
            Qphi_val = Qphi(phi, f, M, Mpl, g, Cg, c, m)[0]
            if epsilon_v(phi,f,M,Mpl)/(1 + Qphi_val) <= 1:
                Q_list.append(Qphi_val)
            else:
                Q_list.append(np.nan)
        except RuntimeError:
            Q_list.append(np.nan)
    Q_list = np.array(Q_list)

    # 2. Encontrar phi_end
    phi_end, _ = fin_inflacion(phi_val, Q_list, f, M, Mpl, g, Cg, c, m)

    # 3. Integrar hacia atrás para N e-folds
    sol = solve_ivp(
        fun=dphi_dN,
        t_span=(0, -N),
        y0=[phi_end],
        args=(f, M, Mpl, g, Cg, c, m),
        method='RK45', #RK89 #pd89
        rtol=1e-8, atol=1e-10
    )

    phi_N = sol.y[0]
    N_vals = sol.t  # t es negativo: 0 -> -N
    N_vals = -N_vals  # invertimos para que N crezca hacia adelante

    # 4. Calcular Q(N) y T/H(N)
    Q_N = np.array([Qphi(phi, f, M, Mpl, g, Cg, c, m)[0] for phi in phi_N])
    H_N = np.array([Hubble(phi, f, M, Mpl) for phi in phi_N])
    T_N = np.array([temperatura(phi, Q, f, M, Mpl, g) for phi, Q in zip(phi_N, Q_N)])
    TH_N = T_N / H_N

    return N_vals, phi_N, Q_N, TH_N

def find_M(f, Cg):
    def objetivo(M):
        #M = np.exp(logM)
        N_vals, phi_N, Q_N, TH_N = background_full(f,M,Cg)
        phi_star = phi_N[-1]
        Q_star   = Q_N[-1]
        G3 = 1+4.981*Q_star**1.946+0.127*Q_star**4.33
        As = espectro_potencias(phi_star, Q_star, f, M, Mpl, g, G3)
        As_obs = 2.10*10**-9
        return As-As_obs
        #return np.log(As) - np.log(As_obs)
    
    sol = root_scalar(objetivo, bracket=[1e-10,0.1])
    return sol.root

F = np.array([5])
#Cgamma = np.linspace(10, 1e7, 150)
Cgamma = np.array([1e4])

Mpl = 1
g = 100
c = 3
m = 0

NS_ = np.full((len(F), len(Cgamma)), np.nan)
Q_star_arr = np.full((len(F), len(Cgamma)), np.nan)
M_arr = np.full((len(F), len(Cgamma)), np.nan)
Pt_arr = np.full((len(F), len(Cgamma)), np.nan)
r_arr = np.full((len(F), len(Cgamma)), np.nan)

for i, f in enumerate(F):
    
    print("iter f: ", i)
    
    for j, Cg in enumerate(Cgamma):
        
        print("iter Cg: ", j)
        
        try:
        
            M = find_M(f,Cg)
            
            M_arr[i,j] = M
            
            N_vals, phi_N, Q_N, TH_N = background_full(f,M,Cg)
            
            phi_star = phi_N[-1]
            Q_star   = Q_N[-1]
            
            Q_star_arr[i,j] = Q_star
            
            G3 = 1+4.981*Q_star**1.946+0.127*Q_star**4.33
            dlnG3 = ((4.981*1.946)*Q_star**0.946 + (0.127*4.33)*Q_star**3.33)/G3
            
            pt = espectro_tensorial(phi_star,f,M,Mpl)
            r = pt/espectro_potencias(phi_star,Q_star,f,M,Mpl,g, G3)
            Pt_arr[i,j] = pt
            r_arr[i,j] = r
            

            ns = indice_espectral(phi_star, Q_star,f,M,Mpl,g, c, m,dlnG3)
            
            NS[i,j] = ns
            
        except (UnboundLocalError, RuntimeError):

            M_arr[i,j] = np.nan
            Q_star_arr[i,j] = np.nan
            NS[i,j] = np.nan
            Pt_arr[i,j] = np.nan
            r_arr[i,j] = np.nan
            
            

fig, axs = plt.subplots(3,1, figsize=(8,12), sharex=True)

# phi(N)
axs[0].plot(N_vals, phi_N)
axs[0].set_ylabel(r'$\phi(N)$')
axs[0].grid(True)

# Q(N)
axs[1].semilogy(N_vals, Q_N)
axs[1].set_ylabel(r'$Q(N)$')
axs[1].grid(True)

# T/H(N)
axs[2].semilogy(N_vals, TH_N)
axs[2].set_ylabel(r'$T/H(N)$')
axs[2].set_xlabel("Número de e-folds $N$")
axs[2].grid(True)

plt.tight_layout()
plt.show()

plt.semilogx(Q_star_arr_10, Pt_arr)
plt.show()

plt.semilogx(Q_star_arr_10, r_arr)
plt.show()

print("r : ", r_arr)
