import numpy as np
from classy import Class
import os
from scipy.optimize import root
from scipy.interpolate import CubicSpline
from cobaya.theory import Theory
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

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
    #return 1/(np.exp(H/T)-1)
    return 1/(np.expm1(H/T))

def Aphi(phi,f,M,Mpl,g, Cg, c,m):
    H = Hubble(phi,f,M,Mpl)
    V = pot(phi,f,M)
    dV = dpot(phi,f,M)

    X = 15*Mpl**2*dV**2/(2*np.pi**2*g*V)

    A = Cg * X**(c/4) * phi**m /(3*H)

    return A

def Qphi(phi, f, M, Mpl, g, Cg, c, m):

    A = Aphi(phi, f, M, Mpl, g, Cg, c, m)

    if not np.isfinite(A):
        raise ValueError(f"A no finito: {A}")

    if A <= 0:
        return 0.0

    F = lambda Q: f_q(Q, c) - A

    # régimen débil
    Qmin = 1e-50

    # régimen fuerte
    Qmax = max(10.0, A**2)

    while F(Qmax) < 0:
        Qmax *= 10

        if Qmax > 1e100:
            raise RuntimeError("No se pudo acotar la raíz")

    sol = root_scalar(
        F,
        bracket=[Qmin, Qmax],
        method="brentq"
    )

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
    phi, lnQ = Y
    Q = np.exp(lnQ)
    H = Hubble(phi, f, M, Mpl)
    dphi = -dpot(phi,f,M) / (3*H**2 * (1 + Q))
    dlnQ = dlnQdN(phi, Q, f, M, Mpl, c, m)
    return [dphi, dlnQ]

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

def end_system(X, f, M, Mpl, g, Cg, c, m):

    phi, Q = X

    eq1 = epsilon_v(phi,f,M,Mpl)/(1+Q) - 1

    eq2 = f_q(Q,c) - Aphi(phi,f,M,Mpl,g,Cg,c,m)

    return [eq1, eq2]


class WarmNaturalInflation(Theory):

    def initialize(self):

        self.Mpl = 1.0
        self.g = 106.75

        self.c = 3
        self.m = 0

        self._Pk_spline = None
        self._Pk_t_spline = None

        self.last_phi_end = None
        self.last_Q_end = None

    def get_requirements(self):
        return {
        "f": None,
        "logCg": None,
        "M": None,
        }

    def get_can_provide_params(self):
        return [
        "As",
        "ns",
        "r",
        "Q_star",
        "phi_star"
        ]

    def get_primordial_scalar_pk(self):
        return self.current_state["primordial_scalar_pk"]

    def get_primordial_tensor_pk(self):
        return self.current_state["primordial_tensor_pk"]

    def get_can_provide(self):
        return [
        "primordial_scalar_pk",
        "primordial_tensor_pk"
        ]

    def calculate(self, state, want_derived=True, **params):

        f = params["f"]
        logCg = params["logCg"]

        Cg = 10**logCg

        M = params["M"]

        try:

            if self.last_phi_end is None:

                x0 = [0.5*np.pi*f, 1]

            else:

                x0 = [self.last_phi_end, self.last_Q_end]

            sol = root(
                end_system,
                x0=x0,
                args=(f,M,self.Mpl,self.g,Cg,self.c,self.m)
            )

            if not sol.success:
                raise ValueError(sol.message)

            phi_end, Q_end = sol.x
            self.last_phi_end = phi_end
            self.last_Q_end = Q_end

            sol = solve_ivp(
                fun=dYdN,
                t_span=[0, -60],
                y0=[phi_end, np.log(Q_end)],
                args=(
                    f,
                    M,
                    self.Mpl,
                    self.g,
                    Cg,
                    self.c,
                    self.m
                ),
                method = 'DOP853',
                rtol=1e-8, atol=1e-10
            )

            phi_N = sol.y[0][::-1]
            Q_N = np.exp(sol.y[1][::-1])
            N_vals = sol.t[::-1] + 60

            phi_star = phi_N[0]
            Q_star = Q_N[0]

            As = espectro_potencias(
                phi_star,
                Q_star,
                f,
                M,
                self.Mpl,
                self.g,
                self.c
            )

            Ps = espectro_potencias(phi_N,Q_N,f,M,self.Mpl,self.g,self.c)

            ns = indice_espectral(
                phi_star,
                Q_star,
                f,
                M,
                self.Mpl,
                self.g,
                self.c,
                self.m
            )

            At = espectro_tensorial(
                phi_star,
                f,
                M,
                self.Mpl
            )

            Pt = espectro_tensorial(phi_N,f,M,self.Mpl)

            r = At / As

            H_N = Hubble(phi_N, f, M, self.Mpl)
            H_star = H_N[0]

            N_star = N_vals[0]

            k_pivot = 0.05

            k_ar = k_pivot * (np.exp(N_vals)*H_N/(np.exp(N_star) * H_star))

            idx = np.argsort(k_ar)
            k_ar = k_ar[idx]
            Ps = Ps[idx]
            Pt = Pt[idx]

            kmin = 1e-6
            kmax = 100.0

            mask = (
                (k_ar >= kmin)
                & (k_ar <= kmax)
                & np.isfinite(Ps)
                & (k_ar > 0)
                & (Ps > 0)
            )

            mask2 = (
                (k_ar >= kmin)
                & (k_ar <= kmax)
                & np.isfinite(Pt)
                & (k_ar > 0)
                & (Pt > 0)
            )

            k_ar1 = k_ar[mask]
            Ps = Ps[mask]
            k_ar2 = k_ar[mask2]
            Pt = Pt[mask2]

            if (
                not np.isfinite(As)
                or not np.isfinite(ns)
                or not np.isfinite(r)
            ):
                raise ValueError

            if (not np.all(np.isfinite(Ps)) or not np.all(np.isfinite(Pt))):
                raise ValueError("Non-finite spectrum")

            state["logp"] = 0
            state["primordial_scalar_pk"] = {
                "k": k_ar1,
                "Pk": Ps,
                "log_regular": False
            }

            state["primordial_tensor_pk"] = {
                "k": k_ar2,
                "Pk": Pt,
                "log_regular": False
            }

            state["derived"] = {
                "As": float(As),
                "ns": float(ns),
                "r": float(r),
                "Q_star": float(Q_star),
                "phi_star": float(phi_star),
            }

        except Exception:

            state["logp"] = -1e50
            state["derived"] = {
                "As": None,
                "ns": None,
                "r": None,
                "Q_star": None,
                "phi_star": None,
            }
            raise
