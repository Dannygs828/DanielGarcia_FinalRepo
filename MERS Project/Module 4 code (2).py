import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from main_functions import convert_cumulative_to_SIR, euler_sir

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------
csv_path = r"MERS_Saudi_Arabia_data_2013_2014_new_cases.csv"

# Population of Saudi Arabia
N = 28000000  # 28 million people

# Assumed infectious period (weeks someone remains infectious)
infectious_period_weeks = 2

print("Loading data...")

# ------------------------------------------------------------
# LOAD RAW DATA
# ------------------------------------------------------------
df = pd.read_csv(csv_path)

# The CSV has "confirmed_cases" which are NEW cases per week, not cumulative
# So we need to create cumulative cases first
df["cumulative_cases"] = df["confirmed_cases"].cumsum()

date_col_name = "date"
df[date_col_name] = pd.to_datetime(df[date_col_name])
df = df.sort_values(date_col_name).reset_index(drop=True)

# ------------------------------------------------------------
# BUILD S_est, I_est, R_est FROM DATA
# ------------------------------------------------------------
df_full = convert_cumulative_to_SIR(
    df,
    date_col=date_col_name,
    cumulative_col="cumulative_cases",
    population=N,
    infectious_period=infectious_period_weeks,
    I_col="I_est",
    R_col="R_est",
    S_col="S_est"
)

print("\nProcessed data:")
print(df_full[['date', 'confirmed_cases', 'cumulative_cases', 'I_est', 'R_est', 'S_est']].head(10))

# True I(t) from the data:
I_true = df_full["I_est"].values.astype(float)
t_obs = np.arange(len(I_true))    # 0, 1, 2, ... weeks

# Initial conditions from data:
I0 = I_true[0]
R0 = df_full["R_est"].iloc[0]
S0 = N - I0 - R0   # enforce S + I + R = N

print(f"\nInitial conditions: S0={S0:.1f}, I0={I0:.1f}, R0={R0:.1f}")
print(f"I_true range: min={I_true.min():.1f}, max={I_true.max():.1f}")
print(f"Number of time points: {len(t_obs)} weeks")

# ------------------------------------------------------------
# STEP 2: QUICK DEMO WITH TWO PARAMETER GUESSES
# ------------------------------------------------------------
# Use much smaller beta values for this small outbreak
beta1, gamma1 = 0.5, 0.4
beta2, gamma2 = 0.8, 0.6

# Step 1: Get S(t), I(t), R(t) using Euler's method
S1, I1, R1 = euler_sir(beta1, gamma1, S0, I0, R0, t_obs, N)
S2, I2, R2 = euler_sir(beta2, gamma2, S0, I0, R0, t_obs, N)

print(f"\nModel 1 (beta={beta1}, gamma={gamma1}):")
print(f"  I1 range: min={I1.min():.1f}, max={I1.max():.1f}")
print(f"\nModel 2 (beta={beta2}, gamma={gamma2}):")
print(f"  I2 range: min={I2.min():.1f}, max={I2.max():.1f}")

# Step 3: Plot model predictions and true I(t)
plt.figure(figsize=(10, 6))
plt.plot(t_obs, I_true, 'o', label="True I(t) from data", markersize=8, color='black', zorder=3)
plt.plot(t_obs, I1, '-x', label=f"Model I(t), beta={beta1}, gamma={gamma1}")
plt.plot(t_obs, I2, '-s', label=f"Model I(t), beta={beta2}, gamma={gamma2}")
plt.xlabel("Weeks")
plt.ylabel("Infections")
plt.title("Effect of beta, gamma on I(t)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Step 4: Calculate SSE between model and true I(t)
print("SSE for (beta1,gamma1):", np.sum((I1 - I_true)**2))
print("SSE for (beta2,gamma2):", np.sum((I2 - I_true)**2))

# ------------------------------------------------------------
# DEFINE SSE HELPER
# ------------------------------------------------------------
def compute_sse(beta, gamma):
    S_mod, I_mod, R_mod = euler_sir(beta, gamma, S0, I0, R0, t_obs, N)
    return np.sum((I_mod - I_true)**2)


# ------------------------------------------------------------
# STEP 5: OPTIMIZE beta AND gamma TO MINIMIZE SSE
# ------------------------------------------------------------
print("\nOptimizing parameters using scipy.optimize.minimize...")

# Objective function for scipy
def objective(params):
    beta, gamma = params
    if beta <= 0 or gamma <= 0:  # Keep parameters positive
        return 1e10
    return compute_sse(beta, gamma)

# Try multiple starting points to avoid local minima
best_result = None
best_sse_overall = np.inf

starting_points = [
    [0.5, 0.3],
    [1.0, 0.5],
    [0.1, 0.1],
    [2.0, 1.0],
    [0.05, 0.5]
]

for start in starting_points:
    result = minimize(objective, start, method='Nelder-Mead', 
                     options={'maxiter': 1000})
    if result.fun < best_sse_overall:
        best_sse_overall = result.fun
        best_result = result
        print(f"  New best from start {start}: beta={result.x[0]:.4f}, gamma={result.x[1]:.4f}, SSE={result.fun:.2f}")

best_beta, best_gamma = best_result.x
best_sse = best_result.fun

print("\nBest-fit parameters from optimization:")
print(f"  beta  = {best_beta:.4f}")
print(f"  gamma = {best_gamma:.4f}")
print(f"  R0    = {best_beta/best_gamma:.4f}")
print(f"  SSE   = {best_sse:.4f}")

# ------------------------------------------------------------
# PLOT BEST-FIT MODEL VS TRUE I(t)
# ------------------------------------------------------------
S_best, I_best, R_best = euler_sir(best_beta, best_gamma, S0, I0, R0, t_obs, N)

plt.figure(figsize=(10, 6))
plt.plot(t_obs, I_true, 'o', label="True I(t) from data")
plt.plot(t_obs, I_best, '-', label=f"Best model I(t)\nβ={best_beta:.3f}, γ={best_gamma:.3f}")
plt.xlabel("Weeks")
plt.ylabel("Infections")
plt.title("Best-Fit SIR Model vs Data (I(t))")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# PLOT S, I, R FOR BEST-FIT MODEL
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_obs, S_best, label="S model")
plt.plot(t_obs, I_best, label="I model")
plt.plot(t_obs, R_best, label="R model")
plt.xlabel("Weeks")
plt.ylabel("Number of people")
plt.title("Best-Fit SIR Model: S(t), I(t), R(t)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# STEP 2: FIRST-HALF FIT → PREDICT SECOND HALF
# ------------------------------------------------------------
print("\n--- STEP 2: First-Half Training, Second-Half Prediction ---")

# 1. Split data into halves
mid = len(I_true) // 2
t_first = t_obs[:mid]
I_first = I_true[:mid]
t_second = t_obs[mid:]
I_second = I_true[mid:]

# 2. Initial conditions for first half
I0_half = I_first[0]
R0_half = df_full["R_est"].iloc[0]
S0_half = N - I0_half - R0_half

# 3. SSE function on first half
def compute_sse_first_half(beta, gamma):
    S_mod, I_mod, R_mod = euler_sir(beta, gamma, S0_half, I0_half, R0_half, t_first, N)
    return np.sum((I_mod - I_first)**2)

def objective_half(params):
    beta, gamma = params
    if beta <= 0 or gamma <= 0:
        return 1e10
    return compute_sse_first_half(beta, gamma)

# 4. Optimize beta/gamma on first half
best_half_result = None
best_half_sse = np.inf
for start in starting_points:
    result = minimize(objective_half, start, method='Nelder-Mead', options={'maxiter': 1000})
    if result.fun < best_half_sse:
        best_half_sse = result.fun
        best_half_result = result
        print(f"  New best (first half) from start {start}: beta={result.x[0]:.4f}, gamma={result.x[1]:.4f}, SSE={result.fun:.2f}")

best_beta_half, best_gamma_half = best_half_result.x
print("\nFirst-half best-fit parameters:")
print(f"  beta_half  = {best_beta_half:.4f}")
print(f"  gamma_half = {best_gamma_half:.4f}")
print(f"  R0_half    = {best_beta_half / best_gamma_half:.4f}")
print(f"  SSE_half   = {best_half_sse:.2f}")

# 5. Simulate FIRST HALF using best-half parameters
S_first_fit, I_first_fit, R_first_fit = euler_sir(
    best_beta_half, best_gamma_half,
    S0_half, I0_half, R0_half,
    t_first, N
)

# 6. Get state at split point to continue simulation
S_split, I_split, R_split = S_first_fit[-1], I_first_fit[-1], R_first_fit[-1]

# 7. Simulate SECOND HALF starting from split point
S_second_fit, I_second_fit, R_second_fit = euler_sir(
    best_beta_half, best_gamma_half,
    S_split, I_split, R_split,
    t_second, N
)

# 8. Concatenate first + second half for plotting
S_pred = np.concatenate([S_first_fit, S_second_fit])
I_pred = np.concatenate([I_first_fit, I_second_fit])
R_pred = np.concatenate([R_first_fit, R_second_fit])

# 9. SSE on SECOND HALF only
SSE_second_half = np.sum((I_second_fit - I_second)**2)
print(f"\nSSE on SECOND HALF of data (using first-half parameters): {SSE_second_half:.2f}")

# 10. Plot predictions vs actual data
plt.figure(figsize=(10, 6))
plt.plot(t_obs, I_true, 'o', label="True I(t)", color="black")
plt.plot(t_obs, I_pred, '-', label="Prediction using First-Half Fit", color='orange')
plt.axvline(mid, color='red', linestyle='--', label="Train/Test Split")
plt.xlabel("Weeks")
plt.ylabel("Infections")
plt.title("Step 2: First-Half Fit → Predict Second Half")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

from scipy.integrate import solve_ivp

# ------------------------------------------------------------
# RK4 version of SIR simulation using solve_ivp
# ------------------------------------------------------------
def sir_rhs(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def rk4_sir(beta, gamma, S0, I0, R0, t, N):
    # solve_ivp returns interpolated values; we evaluate at t points
    sol = solve_ivp(
        sir_rhs,
        t_span=(t[0], t[-1]),
        y0=[S0, I0, R0],
        args=(beta, gamma, N),
        t_eval=t,
        method='RK45'  # Runge-Kutta 4(5)
    )
    S, I, R = sol.y
    return S, I, R

# ------------------------------------------------------------
# Define SSE helper for RK4
# ------------------------------------------------------------
def compute_sse_rk4(beta, gamma, S0, I0, R0, t_obs, I_true):
    S_mod, I_mod, R_mod = rk4_sir(beta, gamma, S0, I0, R0, t_obs, N)
    return np.sum((I_mod - I_true)**2)

# ------------------------------------------------------------
# Optimize beta and gamma using RK4
# ------------------------------------------------------------
best_result_rk4 = None
best_sse_rk4 = np.inf

starting_points = [
    [0.5, 0.3],
    [1.0, 0.5],
    [0.1, 0.1],
    [2.0, 1.0],
    [0.05, 0.5]
]

for start in starting_points:
    result = minimize(
        lambda params: compute_sse_rk4(params[0], params[1], S0, I0, R0, t_obs, I_true),
        start,
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )
    if result.fun < best_sse_rk4:
        best_sse_rk4 = result.fun
        best_result_rk4 = result
        print(f"  New RK4 best from start {start}: "
              f"beta={result.x[0]:.4f}, gamma={result.x[1]:.4f}, SSE={result.fun:.2f}")

best_beta_rk4, best_gamma_rk4 = best_result_rk4.x
S_rk4, I_rk4, R_rk4 = rk4_sir(best_beta_rk4, best_gamma_rk4, S0, I0, R0, t_obs, N)

print("\nRK4 Best-fit parameters:")
print(f"  beta  = {best_beta_rk4:.4f}")
print(f"  gamma = {best_gamma_rk4:.4f}")
print(f"  R0    = {best_beta_rk4 / best_gamma_rk4:.4f}")
print(f"  SSE   = {best_sse_rk4:.4f}")

# ------------------------------------------------------------
# Plot RK4 vs data
# ------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(t_obs, I_true, 'o', label="True I(t)", color="black")
plt.plot(t_obs, I_rk4, '-', label=f"RK4 Best-fit I(t)\nβ={best_beta_rk4:.3f}, γ={best_gamma_rk4:.3f}", color="orange")
plt.xlabel("Weeks")
plt.ylabel("Infections")
plt.title("RK4 SIR Model Fit vs Data")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
