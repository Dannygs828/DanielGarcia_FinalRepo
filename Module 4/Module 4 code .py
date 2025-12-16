import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from main_functions import convert_cumulative_to_SIR, euler_sir

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------
csv_path = r"MERS_Saudi_Arabia_data_2013_2014_new_cases.csv"

# Population in millions (same units as your data)
N = 353000000
# Assumed infectious period used to build I_est and R_est from data
infectious_period_days = 2 

# ------------------------------------------------------------
# LOAD RAW DATA
# ------------------------------------------------------------
df = pd.read_csv(csv_path)
df["cumulative_cases"] = df["confirmed_cases"].cumsum()
# Adjust these names if your CSV uses different column names
date_col_name = "date"


df[date_col_name] = pd.to_datetime(df[date_col_name])
df = df.sort_values(date_col_name)

# ------------------------------------------------------------
# BUILD S_est, I_est, R_est FROM DATA
# (this uses the helper your instructor gave you)
# ------------------------------------------------------------
df_full = convert_cumulative_to_SIR(
    df,
    date_col=date_col_name,
    cumulative_col= "cumulative_cases",
    population=N,
    infectious_period=infectious_period_days,
    I_col="I_est",
    R_col="R_est",
    S_col="S_est"
)

print(df_full.head())

# True I(t) from the data:
I_true = df_full["I_est"].values.astype(float)
t_obs = np.arange(len(I_true))    # 0,1,2,... days

# Initial conditions from data:
I0 = I_true[0]
R0 = df_full["R_est"].iloc[0]
S0 = N - I0 - R0   # enforce S + I + R = N

print(f"Initial conditions: S0={S0:.4f}, I0={I0:.4f}, R0={R0:.4f}")

# ------------------------------------------------------------
# QUICK DEMO: TWO PARAMETER GUESSES
# ------------------------------------------------------------
beta1, gamma1 = .001, .001
beta2, gamma2 = 0.9, .8

S1, I1, R1 = euler_sir(beta1, gamma1, S0, I0, R0, t_obs, N)
S2, I2, R2 = euler_sir(beta2, gamma2, S0, I0, R0, t_obs, N)

plt.figure(figsize=(10, 6))
plt.plot(t_obs, I_true, 'o', label="True I(t) from data")
plt.plot(t_obs, I1, '-x', label=f"Model I(t), beta={beta1}, gamma={gamma1}")
plt.plot(t_obs, I2, '-s', label=f"Model I(t), beta={beta2}, gamma={gamma2}")
plt.xlabel("Days")
plt.ylabel("Infections (millions)")
plt.title("Effect of beta, gamma on I(t)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("SSE for (beta1,gamma1):", np.mean((I1 - I_true)**2))
print("SSE for (beta2,gamma2):", np.mean((I2 - I_true)**2))

# ------------------------------------------------------------
# DEFINE SSE HELPER
# ------------------------------------------------------------
def compute_sse(beta, gamma):
    S_mod, I_mod, R_mod = euler_sir(beta, gamma, S0, I0, R0, t_obs, N)
    return np.sum((I_mod - I_true)**2)


# ------------------------------------------------------------
# GRID SEARCH OVER beta AND gamma
# ------------------------------------------------------------
# beta_vals = np.linspace(0.05, 0.8, 30)           # try 30 values between 0.05 and 0.8
# gamma_vals = np.linspace(1/21, 1/2, 30)          # recovery periods ~2–21 days

# best_sse = np.inf
# best_beta = None
# best_gamma = None

# for beta in beta_vals:
#     for gamma in gamma_vals:
#         sse = compute_sse(beta, gamma)
#         if sse < best_sse:
#             best_sse = sse
#             best_beta, best_gamma = beta, gamma

# print("\nBest-fit parameters from grid search:")
# print(f"  beta  = {best_beta:.4f}")
# print(f"  gamma = {best_gamma:.4f}")
# print(f"  SSE   = {best_sse:.4f}")
best_beta = 1/2
best_gamma= 1/2
# ------------------------------------------------------------
# PLOT BEST-FIT MODEL VS TRUE I(t)
# ------------------------------------------------------------
S_best, I_best, R_best = euler_sir(best_beta, best_gamma, S0, I0, R0, t_obs, N)

plt.figure(figsize=(10, 6))
plt.plot(t_obs, I_true, 'o', label="True I(t) from data")
plt.plot(t_obs, I_best, '-', label=f"Best model I(t)\nβ={best_beta:.3f}, γ={best_gamma:.3f}")
plt.xlabel("Days")
plt.ylabel("Infections (millions)")
plt.title("Best-Fit SIR Model vs Data (I(t))")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# OPTIONAL: PLOT S, I, R FOR BEST-FIT MODEL
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_obs, S_best, label="S model")
plt.plot(t_obs, I_best, label="I model")
plt.plot(t_obs, R_best, label="R model")
plt.xlabel("Days")
plt.ylabel("Millions of people")
plt.title("Best-Fit SIR Model: S(t), I(t), R(t)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print(df_full.head())