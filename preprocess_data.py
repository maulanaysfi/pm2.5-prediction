import pandas as pd

df = pd.read_csv('/home/ibrahim/jupyter/models/pm2_predict/City_Types.csv')

df['Type'] = df['Type'].astype('category')

def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    print(f'{column} IQR value: {IQR}')

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df[column]

print(f'Dataset size before outliers handled: {df.size}')
print(f'Dataset NaN values before outliers handled: {df.isna().sum().sum()}\n')

df['CO'] = handle_outliers(df, 'CO')
df['SO2'] = handle_outliers(df, 'SO2')
df['NO2'] = handle_outliers(df, 'NO2')
df['O3'] = handle_outliers(df, 'O3')

print(f'\nDataset NaN values after outliers handled: {df.isna().sum().sum()}')

print('Removing NaN values...', end='')
df.dropna(inplace=True)
print(f'Dataset size after outliers handled: {df.size}')

filename = "data_preprocessed.csv"
print(f'Exporting to CSV ({filename})...', end='')
df.to_csv(filename, index=False)
print('Done!')