# Libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

st.title("📊 Comparador de Modelos de Regressão (Versão Oculta)")

# Upload de arquivo CSV
#uploaded_file = st.file_uploader("Envie o arquivo CSV com os dados de imóveis:", type=["csv"])
uploaded_file = "sao-paulo-properties-april-2019.csv"
if uploaded_file is not None:
    # --- TRATAMENTO DE DADOS ---
    df_data = pd.read_csv(uploaded_file)

    st.write("### Prévia dos Dados")
    st.dataframe(df_data.head())

    # Filtra apenas imóveis para aluguel
    df_rent = df_data[df_data['Negotiation Type'] == 'rent']
    df_cleaned = df_rent.drop(['New', 'Property Type', 'Negotiation Type'], axis=1)

    # One-Hot Encoding
    one_hot = pd.get_dummies(df_cleaned['District']).astype(int)
    df = df_cleaned.drop("District", axis=1)
    df = df.join(one_hot)

    # Definir X e Y
    Y = df['Price']
    X = df.drop(columns=['Price'])

    # --- TREINO E TESTE ---
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # --- GRID SEARCH (OCULTO) ---
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    model = RandomForestRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)

    # --- RESULTADOS ---
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # --- EXIBIÇÃO ---
    st.subheader("Resultados do Modelo Oculto")
    col1, col2, col3 = st.columns(3)
    col1.metric("Erro Quadrático Médio (MSE)", f"{mse:.2f}")
    col2.metric("Raiz do Erro Quadrático Médio (RMSE)", f"{rmse:.2f}")
    col3.metric("Número de features", X.shape[1])

    st.write("### Estatísticas do Grid Search")
    results = pd.DataFrame(grid_search.cv_results_)
    st.dataframe(results[["mean_test_score", "std_test_score", "params"]])

else:
    st.info("Por favor, envie um arquivo CSV para iniciar o processamento.")
