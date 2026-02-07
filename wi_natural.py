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
    #x = x0
    
    #for i in range(iter):
    #    xn = x - f/df
    #    
    #    if abs(xn - x) < tol:
    #        return xn
        
    #    x = xn
        
    raise RuntimeError("No se alcanzó convergencia")

####    CALCULO DE Q(PHI)    ####

def Qphi(phi,f,M,Mpl,g, Cg, c,m):
    
    
    A = Aphi(phi,f,M,Mpl,g, Cg, c,m)
        
    Q = A**(4/(4-c))
        
    fq = f_q(Q,c)
    df = dfq(Q,c)
        
    fw = weak_fq(Q,c)
    dfw = weak_dfq(Q,c)
        
    fs = strong_fq(Q,c)
    dfs = 1
        
    #Fq = fq-A
    #Fw = fw - A
    #Fs = fs - A
    
    Fq = lambda Q: f_q(Q,c)-A
    df = lambda Q: dfq(Q,c)
        
    Q = newton(Fq,df,Q, c,tol=1e-3,iter=1000)
    Qw = 0 #newton(Fq,dfq,Q, c,tol=1e-5,iter=1000)
    #Qs = newton(Fs,dfs,Qs0, tol=1e-5,iter=1000000)
    
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
    #G3 = float(1+4.981*Q**1.946+0.127*Q**4.33)
    #dlnG3 = float(((4.981*1.946)*Q**0.946 + (0.127*4.33)*Q**3.33)/G3)
    B = 1+2*n + (2*Q*T*np.pi*np.sqrt(3))/(H*np.sqrt(3+4*np.pi*Q))
    
    corch = 2*Q*T*np.sqrt(3)*(dlnQ+dlnTH-4*np.pi*Q*dlnQ/(4+4*np.pi*Q))/(H*np.sqrt(3+4*np.pi*Q))-(2*n*dlnTH*(T/H)/(1-np.exp(-H/T)))

    return 1-6*ev+2*etv+Q*dlnQ/(1+Q)+Q*dlnQ*dlnGi+corch/B

def espectro_tensorial(phi,f,M,Mpl):
    H = Hubble(phi,f,M,Mpl)
    
    return 8* (H/(2*np.pi))**2/Mpl**2

def background(f, M, Cg, Mpl=1, g=100, c=3, m=-2, N=60):
    phi_val = np.linspace(0, np.pi*f,50000)

    Q = []

    for phi in phi_val:
            try:
                if epsilon_v(phi,f,M,Mpl)/(1+Qphi(phi,f,M,Mpl,g, Cg, c,m)[0]) <= 1.0:
                    Q.append(Qphi(phi,f,M,Mpl,g, Cg, c,m)[0])
                else:
                    Q.append(np.nan)
            except RuntimeError:
                Q.append(np.nan)

    phi_end = fin_inflacion(phi_val, Q, f, M,Mpl,g,Cg,c,m)[0]

    sol = solve_ivp(
    fun=dphi_dN,
    t_span=(0, -N),    # integración hacia atrás
    y0=[phi_end],            # valor inicial de phi
    args=(f, M, Mpl, g, Cg, c, m),
    method='RK45',           # Runge-Kutta 4(5)
    rtol=1e-8, atol=1e-10)

    phi_star = sol.y[0, -1]  # valor de phi al inicio de inflación  

    Q_star = Qphi(phi_star,f,M,Mpl,g,Cg,c,m)[0]
    
    return phi_star, Q_star

def find_M(f, Cg):
    def objetivo(M):
        #M = np.exp(logM)
        phi_star, Q_star = background(f,M,Cg)
        G3 = 1+4.981*Q_star**1.946+0.127*Q_star**4.33
        As = espectro_potencias(phi_star, Q_star, f, M, Mpl, g, G3)
        As_obs = 2.10*10**-9
        return As-As_obs
        #return np.log(As) - np.log(As_obs)
    
    sol = root_scalar(objetivo, bracket=[1e-10,0.1])
    return sol.root

F = np.array([2,3,4,5,10])
Cgamma = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12])

Mpl = 1
g = 100
c = 3
m = 0

NS = np.full((len(F), len(Cgamma)), np.nan)
Q_star_arr = np.full((len(F), len(Cgamma)), np.nan)
M_arr = np.full((len(F), len(Cgamma)), np.nan)

for i, f in enumerate(F):
    
    print("iter f: ", i)
    
    for j, Cg in enumerate(Cgamma):
        
        print("iter Cg: ", j)
        
        try:
        
            M = find_M(f,Cg)
            
            M_arr[i,j] = M
            
            phi_star, Q_star = background(f,M,Cg)
            
            Q_star_arr[i,j] = Q_star
            
            G3 = 1+4.981*Q_star**1.946+0.127*Q_star**4.33
            dlnG3 = ((4.981*1.946)*Q_star**0.946 + (0.127*4.33)*Q_star**3.33)/G3

            ns = indice_espectral(phi_star, Q_star,f,M,Mpl,g, c, m,dlnG3)
            
            NS[i,j] = ns
            
        except (UnboundLocalError, RuntimeError):

            M_arr[i,j] = np.nan
            Q_star_arr[i,j] = np.nan
            NS[i,j] = np.nan

ns_obs = 0.9678
ns_obs_err = 0.0072

#for j, Cg in enumerate(Cgamma):
        #plt.plot(F, NS[:, j], label=f"Cγ = {Cg:.0e}")

for i, f in enumerate(F):
        plt.semilogx(Q_star_arr[i,:], NS[i,:], label=f"f = {f}")

plt.axhline(ns_obs, linestyle='-' ,color='r', label=r"$n_s^{\rm obs}$")
plt.axhline(ns_obs+ns_obs_err, linestyle='--' ,color='r')
plt.axhline(ns_obs-ns_obs_err, linestyle='--' ,color='r')
plt.xlabel(r"$Q_*$")
plt.ylabel(r"$n_s$")
plt.legend()
plt.show()

print(M_arr[:,0])
print(NS[:,0])
print(Q_star_arr[:,0])
