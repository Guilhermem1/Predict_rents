import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

df_data = pd.read_csv("sao-paulo-properties-april-2019.csv")
df_rent = df_data[df_data['Negotiation Type'] == 'rent']
df_numeric = df_rent.select_dtypes(include=['number'])
df_correl = df_numeric.corr()
df_cleaned = df_rent.drop(['New','Property Type', 'Negotiation Type'], axis=1)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

district_encoded = ordinal_encoder.fit_transform(df_rent[['District']])
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(df_cleaned[['District']])
housing_cat_1hot
housing_cat_1hot.toarray()
one_hot = pd.get_dummies(df_cleaned['District'])
one_hot = one_hot.astype(int)
df = df_cleaned.drop("District", axis=1)
df = df.join(one_hot)
from sklearn.model_selection import train_test_split

# Treine com os dados de X e compare-os com o preço (Y)
Y = df['Price']
X = df.loc[:,df.columns != "Price"]
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3) #Testando com 70% dos dados e verificar dps com os 30%
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train,y_train)

preds = rf_reg.predict(x_train)
rf_mse = mean_squared_error(y_train,preds)

rf_rmse = np.sqrt(rf_mse)

score = cross_val_score(rf_reg,x_train, y_train, scoring='neg_mean_squared_error', cv=10)
rf_rmse_score = np.sqrt(-score)

def display_scores(score):
    print("Scores: ",score)
    print("Mean: ",score.mean())
    print("Standard deviation: ",score.std())

display_scores(rf_rmse_score)

from sklearn.model_selection import GridSearchCV

param_grid = [
        {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
         {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(x_train,y_train)

final_model = grid_search.best_estimator_
final_model_predictions = final_model.predict(x_test)   # Verificando já com os testes (30%)
 
final_mse = mean_squared_error(y_test, final_model_predictions)
print(np.sqrt(final_mse))

fig = go.Figure(data=[go.Scatter(y=y_test.values),
                      go.Scatter(y=final_model_predictions)])

fig.show()










