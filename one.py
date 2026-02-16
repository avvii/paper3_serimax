# ==========================================
# INSTALL (if needed)
# ==========================================
# !pip install statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. METRIC FUNCTION
# ==========================================
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return mse, rmse, mae, mape


# ==========================================
# 2. SEIRM MODEL
# ==========================================
def seirm_deriv(y, t, params):
    S, E, I, R, M = y
    Pi, beta, theta, lam, rho, alpha, gamma, eta, eta0, mu, sigma = params

    force_of_infection = beta * (I + theta * M) * S
    treatment_rate = (rho * I) / (1 + alpha * I)

    dSdt = Pi - force_of_infection - mu * S
    dEdt = force_of_infection - (lam + mu) * E
    dIdt = lam * E - treatment_rate - (gamma + mu + sigma) * I
    dRdt = treatment_rate + gamma * I - mu * R
    dMdt = eta * I - eta0 * M

    return [dSdt, dEdt, dIdt, dRdt, dMdt]


def generate_seirm_covariates(total_days):
    params = [4.4101, 0.00002, 0.1430, 0.0332,
              0.5218, 0.0001, 0.0825, 0.0032,
              0.1, 0.000035, 0.0]

    y0 = [120000.0, 100.0, 50.0, 0.0, 500.0]
    t = np.linspace(0, total_days, total_days)
    solution = odeint(seirm_deriv, y0, t, args=(params,))
    return solution.T[2]  # Infected curve


# ==========================================
# 3. LOAD DATA
# ==========================================
df = pd.read_csv('./data-table.csv')

df['Epi_date_v3'] = pd.to_datetime(df['Epi_date_v3'], format='mixed')
df = df.sort_values('Epi_date_v3')

start_date = pd.to_datetime('2022-06-01')
end_date = pd.to_datetime('2023-02-25')

df = df[(df['Epi_date_v3'] >= start_date) &
        (df['Epi_date_v3'] <= end_date)].copy()

df.set_index("Epi_date_v3", inplace=True)

# Generate SEIRM for full period
seirm_I = generate_seirm_covariates(len(df))

# Train/Test split
split_date = pd.to_datetime('2022-12-17')

train = df.loc[:split_date]
test = df.loc[split_date + pd.Timedelta(days=1):]

y_train = train["Cases"]
y_test = test["Cases"]

n_train = len(train)
n_test = len(test)

I_train = seirm_I[:n_train]
I_test = seirm_I[n_train:]

# ==========================================
# 4. TRAIN SARIMAX
# ==========================================
model_train = SARIMAX(
    np.log1p(y_train),
    order=(1,1,1),
    seasonal_order=(1,1,1,7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results_train = model_train.fit(disp=False)

# Training fitted values
fitted_log = results_train.fittedvalues
fitted_train = np.expm1(fitted_log)
fitted_train = np.maximum(fitted_train, 0)

# ==========================================
# 5. ROLLING ONE-STEP FORECAST
# ==========================================
history = y_train.copy()
rolling_forecast = []

for t in range(len(y_test)):
    
    model_roll = SARIMAX(
        np.log1p(history),
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results_roll = model_roll.fit(disp=False)
    
    forecast_log = results_roll.forecast(steps=1)
    forecast = np.expm1(forecast_log.iloc[0])
    forecast = max(forecast, 0)
    
    rolling_forecast.append(forecast)
    
    history = pd.concat([
        history,
        pd.Series([y_test.iloc[t]], index=[y_test.index[t]])
    ])

rolling_forecast = np.array(rolling_forecast)

# ==========================================
# 6. SCALE SEIRM CURVE (Least Squares)
# ==========================================
Y_train = y_train.values

beta_opt = np.sum(Y_train * I_train) / np.sum(I_train**2)

seirm_train_scaled = I_train * beta_opt
seirm_test_scaled = I_test * beta_opt

# ==========================================
# 7. METRICS
# ==========================================
mse, rmse, mae, mape = calculate_metrics(y_test, rolling_forecast)

print("\n--- FINAL ROLLING SARIMAX MODEL ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# ==========================================
# 8. VISUALIZATION
# ==========================================
plt.figure(figsize=(16,6))

# ----- TRAINING PLOT -----
plt.subplot(1,2,1)

plt.plot(y_train.index, y_train,
         'o:', color='lime', markersize=3,
         label='Ground Truth Training')

plt.plot(y_train.index, fitted_train,
         '-', color='blue', linewidth=2,
         label='SARIMAX Fitted Value')

plt.plot(y_train.index, seirm_train_scaled,
         '--', color='orange', linewidth=2,
         label='SEIRM Curve')

plt.title("SARIMAX - Training Data")
plt.xlabel("Date")
plt.ylabel("Mpox Cases")
plt.legend()
plt.grid(alpha=0.3)

# ----- TEST PLOT -----
plt.subplot(1,2,2)

plt.plot(y_test.index, y_test,
         'o:', color='lime', markersize=4,
         label='Ground Truth Test')

plt.plot(y_test.index, rolling_forecast,
         '-', color='red', linewidth=2,
         label='Rolling Forecast')

plt.plot(y_test.index, seirm_test_scaled,
         '--', color='gold', linewidth=2,
         label='SEIRM Curve')

plt.title("SARIMAX - Test Forecast")
plt.xlabel("Date")
plt.ylabel("Mpox Cases")
plt.legend()
plt.grid(alpha=0.3)

metrics_text = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%"
plt.gcf().text(0.92, 0.5, metrics_text,
               fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

st.pyplot(plt)