import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error
from datetime import datetime

# --- CARREGAMENTO E LIMPEZA ---
df_data = pd.read_csv("sao-paulo-properties-april-2019.csv")
df_rent = df_data[df_data['Negotiation Type'] == 'rent']
df_cleaned = df_rent.drop(['New', 'Property Type', 'Negotiation Type'], axis=1)

# --- ENCODING CATEGÓRICO ---
one_hot = pd.get_dummies(df_cleaned['District'], dtype=int)
df = df_cleaned.drop("District", axis=1)
df = df.join(one_hot)

# --- VARIÁVEIS ---
Y = df['Price']
X = df.loc[:, df.columns != "Price"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# --- TREINAMENTO INICIAL ---
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(x_train, y_train)

# --- AVALIAÇÃO INICIAL ---
preds = rf_reg.predict(x_train)
rf_mse = mean_squared_error(y_train, preds)
rf_rmse = np.sqrt(rf_mse)

score = cross_val_score(rf_reg, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
rf_rmse_score = np.sqrt(-score)

print("Scores (Cross-Validation):", rf_rmse_score)
print("Mean:", rf_rmse_score.mean())
print("Standard deviation:", rf_rmse_score.std())

# --- GRID SEARCH PARA OTIMIZAR ---
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

final_model = grid_search.best_estimator_
final_predictions = final_model.predict(x_test)

final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
final_mae = mean_absolute_error(y_test, final_predictions)

print(f"Melhor RMSE: {final_rmse:.2f}")
print(f"Erro médio absoluto (MAE): {final_mae:.2f}")

# --- GRÁFICO DE COMPARAÇÃO (VALORES REAIS x PREDITOS) ---
comparison_df = pd.DataFrame({
    "Real": y_test.values,
    "Previsto": final_predictions
}).reset_index(drop=True)

fig = go.Figure()

# Linha dos valores reais
fig.add_trace(go.Scatter(
    y=comparison_df["Real"],
    mode="lines",
    name="Valor Real",
    line=dict(width=3)
))

# Linha dos valores previstos
fig.add_trace(go.Scatter(
    y=comparison_df["Previsto"],
    mode="lines",
    name="Valor Previsto",
    line=dict(width=3, dash="dash")
))

fig.update_layout(
    title="Comparação entre valores reais e previstos",
    xaxis_title="Índice da Amostra",
    yaxis_title="Preço do Imóvel",
    template="plotly_white",
    legend=dict(x=0.02, y=0.98)
)

st.plotly_chart(fig, use_container_width=True)

st.metric("RMSE", f"{final_rmse:.2f}")
st.metric("Erro médio absoluto (MAE)", f"{final_mae:.2f}")
st.write(f"Melhores parâmetros: {grid_search.best_params_}")
