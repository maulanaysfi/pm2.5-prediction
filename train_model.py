import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data_preprocessed.csv')

df['Date'] = pd.to_datetime(df['Date'])

features = df[['CO', 'NO2', 'SO2', 'O3']]
label = df[['PM2.5']]

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=44)

model = ElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=1000, tol=1e-05)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" MAE: {mae:.2f}")
print(f" MSE: {mse:.2f}")
print(f" R2 Score: {r2:.2f}\n")

y_pred_df = X_test
y_pred_df['Predicted_PM2.5'] = y_pred
y_pred_df['Observed_PM2.5'] = y_test

fig, axes = plt.subplots(2,2, constrained_layout=True, figsize=(12,8))
axes = axes.flatten()

feature_names = features.columns

with open('metrics.json', 'w') as outfile:
    json.dump({ "mae": mae, "mse": mse, "r2_score": r2 }, outfile)


for i, col in enumerate(feature_names):
    sns.scatterplot(data=y_pred_df, x=col, y='Observed_PM2.5', alpha=0.3, label='Observed Data', ax=axes[i])
    sns.lineplot(data=y_pred_df, x=col, y='Predicted_PM2.5', color='red', label='Model Prediction', ax=axes[i])
    axes[i].set_title(f'{col} vs PM2.5 Scatterplot with Model Prediction')

fig.savefig('scatterplot_with_prediction.png', dpi=100)