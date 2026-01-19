# homogenization_1d.py
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import time

# -------------------------
# Parameters & coefficients
# -------------------------
def a_y(y): return 1.0 + 0.5*np.sin(2*pi*y)
def b_y(y): return 1.0 + 0.3*np.cos(2*pi*y)
def k_y(y): return 1.0 + 0.5*np.sin(2*pi*y)

H = 2.0
def f0(x): return np.sin(pi*x)

eps_list = [1/10, 1/20, 1/40, 1/100]

# Discretization
Nx = 1500            # intervals (increase for more resolution)
x = np.linspace(0,1,Nx+1)
h = x[1]-x[0]
interior = np.arange(1, Nx)  # interior node indices

# -------------------------
# Homogenized constants (1D harmonic mean)
# -------------------------
ys = np.linspace(0,1,5001)
A_h = 1.0 / np.mean(1.0 / a_y(ys))
B_h = 1.0 / np.mean(1.0 / b_y(ys))
k_bar = np.mean(k_y(ys))
print("A_h =", A_h, "B_h =", B_h, "k_bar =", k_bar)

# -------------------------
# Build cell correctors (1D)
# -------------------------
Ys_fine = np.linspace(0,1,2001)
chiA_prime = (A_h / a_y(Ys_fine)) - 1.0
chiB_prime = (B_h / b_y(Ys_fine)) - 1.0
chiA = np.cumsum(chiA_prime) * (Ys_fine[1]-Ys_fine[0])
chiA = chiA - np.mean(chiA)
chiB = np.cumsum(chiB_prime) * (Ys_fine[1]-Ys_fine[0])
chiB = chiB - np.mean(chiB)

def chiA_of_y(y):
    y_mod = np.mod(y,1.0)
    return np.interp(y_mod, Ys_fine, chiA)

def chiB_of_y(y):
    y_mod = np.mod(y,1.0)
    return np.interp(y_mod, Ys_fine, chiB)

# -------------------------
# Utility: central derivative
# -------------------------
def derivative(arr):
    d = np.zeros_like(arr)
    d[interior] = (arr[interior+1] - arr[interior-1])/(2*h)
    d[0] = (arr[1]-arr[0])/h
    d[-1] = (arr[-1]-arr[-2])/h
    return d

# -------------------------
# Solve homogenized problem
# (triangular approach)
# -------------------------
def solve_homogenized(Ah, Bh, kbar, H, f0func):
    N = len(interior)
    # L_A and L_B for constant coefficients (second derivative discretization)
    L_A = np.zeros((N,N)); L_B = np.zeros((N,N))
    for i, idx in enumerate(interior):
        L_A[i,i] = (2*Ah)/(h*h)
        L_B[i,i] = (2*Bh)/(h*h)
        if i>0:
            L_A[i,i-1] = -Ah/(h*h)
            L_B[i,i-1] = -Bh/(h*h)
        if i<N-1:
            L_A[i,i+1] = -Ah/(h*h)
            L_B[i,i+1] = -Bh/(h*h)
    fx = f0func(x[interior])
    # Solve (L_B + kbar*H I) v = kbar * f0
    v_in = np.linalg.solve(L_B + kbar*H*np.eye(N), kbar*fx)
    u_in = np.linalg.solve(L_A, kbar*(H*v_in - fx))
    u0 = np.zeros(len(x)); v0 = np.zeros(len(x))
    u0[interior] = u_in; v0[interior] = v_in
    return u0, v0

u0, v0 = solve_homogenized(A_h, B_h, k_bar, H, f0)
u0_x = derivative(u0); v0_x = derivative(v0)

# -------------------------
# Solve epsilon-problem (finite difference, block system)
# -------------------------
def solve_epsilon_problem(eps):
    N = len(interior)
    y_nodes = x/eps
    a_nodes = a_y(y_nodes); b_nodes = b_y(y_nodes); k_nodes = k_y(y_nodes)
    a_mid = 0.5*(a_nodes[:-1] + a_nodes[1:])
    b_mid = 0.5*(b_nodes[:-1] + b_nodes[1:])
    # Assemble L_A and L_B (interior)
    L_A = np.zeros((N,N)); L_B = np.zeros((N,N))
    for i, idx in enumerate(interior):
        L_A[i,i] = (a_mid[idx-1] + a_mid[idx])/(h*h)
        L_B[i,i] = (b_mid[idx-1] + b_mid[idx])/(h*h)
        if i>0:
            L_A[i,i-1] = -a_mid[idx-1]/(h*h)
            L_B[i,i-1] = -b_mid[idx-1]/(h*h)
        if i<N-1:
            L_A[i,i+1] = -a_mid[idx]/(h*h)
            L_B[i,i+1] = -b_mid[idx]/(h*h)
    Kdiag = k_nodes[interior]
    # Block matrix assemble
    A_block = np.zeros((2*N,2*N))
    A_block[:N,:N] = L_A
    A_block[:N,N:] = np.diag(-Kdiag*H)
    A_block[N:,N:] = L_B + np.diag(Kdiag*H)
    rhs = np.concatenate([Kdiag * f0(x[interior]), Kdiag * f0(x[interior])])
    sol = np.linalg.solve(A_block, rhs)
    u = np.zeros(len(x)); v = np.zeros(len(x))
    u[interior] = sol[:N]; v[interior] = sol[N:]
    return u, v

# -------------------------
# Run experiments & compute errors
# -------------------------
results = []
for eps in eps_list:
    t0 = time.time()
    u_eps, v_eps = solve_epsilon_problem(eps)
    t1 = time.time()
    ux_eps = derivative(u_eps); vx_eps = derivative(v_eps)
    # correctors
    chiA_vals = chiA_of_y(x/eps)
    chiB_vals = chiB_of_y(x/eps)
    u_corr = u0 + eps * chiA_vals * u0_x
    v_corr = v0 + eps * chiB_vals * v0_x
    # integrals approx by sum * h
    L2_u = np.sum((u_eps - u0)**2)*h
    L2_v = np.sum((v_eps - v0)**2)*h
    H1_u = np.sum((ux_eps - u0_x)**2)*h
    H1_v = np.sum((vx_eps - v0_x)**2)*h
    H1_u_corr = np.sum((ux_eps - derivative(u_corr))**2)*h
    H1_v_corr = np.sum((vx_eps - derivative(v_corr))**2)*h
    results.append((eps, t1-t0, L2_u, L2_v, H1_u, H1_v, H1_u_corr, H1_v_corr))
    print(f"eps={eps:.5f}, time={t1-t0:.3f}s, L2_u={L2_u:.3e}, H1_u={H1_u:.3e}, H1_u_corr={H1_u_corr:.3e}")

print("\nResults summary:")
for row in results:
    print(row)

# Plot H1 error vs eps
eps_vals = np.array([r[0] for r in results])
H1_vals = np.array([r[4] for r in results])
H1_corr_vals = np.array([r[6] for r in results])
plt.figure(figsize=(6,4))
plt.loglog(eps_vals, H1_vals, marker='o', label='||∂x u_eps - ∂x u0||^2')
plt.loglog(eps_vals, H1_corr_vals, marker='s', label='||∂x u_eps - ∂x u_corr||^2')
plt.xlabel('epsilon')
plt.ylabel('H1-error (squared)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
