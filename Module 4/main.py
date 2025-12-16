import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize



# ============================================================
# 1. Convert cumulative cases → estimated S, I, R
# ============================================================

def convert_cumulative_to_SIR(df, date_col='date', cumulative_col='cumulative_cases',
                              population=None, infectious_period=8, recovered_col=None,
                              new_case_col='new_cases', I_col='I_est', R_col='R_est', S_col='S_est'):

    df = df.copy()

    # Ensure sorted by date
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # New cases
    df[new_case_col] = df[cumulative_col].diff().fillna(df[cumulative_col].iloc[0])
    df[new_case_col] = df[new_case_col].clip(lower=0)

    # I(t): rolling sum over infectious period
    df[I_col] = df[new_case_col].rolling(window=infectious_period, min_periods=1).sum()

    # R(t)
    if recovered_col and recovered_col in df.columns:
        df[R_col] = df[recovered_col].fillna(0)
    else:
        df[R_col] = df[cumulative_col].shift(infectious_period).fillna(0)

    # S(t)
    if population is not None:
        df[S_col] = population - df[I_col] - df[R_col]
        df[S_col] = df[S_col].clip(lower=0)
    else:
        df[S_col] = np.nan

    return df



# ============================================================
# 2. Correct Euler SIR Solver
# ============================================================

def euler_sir(beta, gamma, S0, I0, R0, t, N):

    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    S[0], I[0], R[0] = S0, I0, R0

    for n in range(len(t)-1):
        dt = t[n+1] - t[n]

        # correct SIR dynamics
        dS = -(beta / N) * S[n] * I[n]
        dI = (beta / N) * S[n] * I[n] - gamma * I[n]
        dR = gamma * I[n]

        # Euler updates
        S[n+1] = S[n] + dS * dt
        I[n+1] = I[n] + dI * dt
        R[n+1] = R[n] + dR * dt

    return S, I, R



# ============================================================
# 3. SSE Loss function (Used for fitting)
# ============================================================

def sir_loss(params, t, S_data, I_data, R_data, N):

    beta, gamma = params
    if beta < 0 or gamma < 0:
        return 1e12  # enforce positivity

    S0, I0, R0 = S_data[0], I_data[0], R_data[0]

    S_model, I_model, R_model = euler_sir(beta, gamma, S0, I0, R0, t, N)

    # **Main loss is I(t)**
    SSE_I = np.sum((I_model - I_data)**2)

    # small penalty for R
    SSE_R = np.sum((R_model - R_data)**2)

    return SSE_I + 0.1 * SSE_R



# ============================================================
# 4. Load Your Data (REPLACE WITH YOUR REAL CSV)
# ============================================================

df = pd.DataFrame({
    "date": pd.date_range("2020-01-01", periods=40),
    "cumulative_cases": np.cumsum(np.random.poisson(5, 40))
})

population = 1_000_000
infectious_period = 8

df_sir = convert_cumulative_to_SIR(
    df,
    date_col="date",
    cumulative_col="cumulative_cases",
    population=population,
    infectious_period=infectious_period
)

# Prepare arrays
t = np.arange(len(df_sir))
S_data = df_sir["S_est"].values
I_data = df_sir["I_est"].values
R_data = df_sir["R_est"].values
N = S_data[0] + I_data[0] + R_data[0]



# ============================================================
# 5. Fit Parameters β, γ
# ============================================================

result = minimize(
    sir_loss,
    x0=[0.3, 0.1],
    args=(t, S_data, I_data, R_data, N),
    bounds=[(0.0001, 3), (0.0001, 3)]
)

beta_fit, gamma_fit = result.x
print("Fitted β =", beta_fit)
print("Fitted γ =", gamma_fit)

# compute model curves
S_model, I_model, R_model = euler_sir(beta_fit, gamma_fit, S_data[0], I_data[0], R_data[0], t, N)



# ============================================================
# 6. Compute Final SSE
# ============================================================

SSE = np.sum((I_model - I_data)**2)
print("Final SSE =", SSE)



# ============================================================
# 7. FIXED PLOTS
# ============================================================

# ----- Observed vs Model I(t) -----
plt.figure(figsize=(10,5))
plt.plot(t, I_data, 'o-', label="Observed I", linewidth=2)
plt.plot(t, I_model, '--', label="Model I (SIR)", linewidth=2)
plt.title("Observed vs Model I(t)")
plt.xlabel("Time (days)")
plt.ylabel("Infected")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# ----- Model SIR curves -----
plt.figure(figsize=(10,5))
plt.plot(t, S_model, label="S(t)")
plt.plot(t, I_model, label="I(t)")
plt.plot(t, R_model, label="R(t)")
plt.title("SIR Model Trajectories (Fitted)")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# ----- Estimated S, I, R from data -----
plt.figure(figsize=(10,5))
plt.plot(t, S_data, label="S_est")
plt.plot(t, I_data, label="I_est")
plt.plot(t, R_data, label="R_est")
plt.title("Estimated S, I, R from Data")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
