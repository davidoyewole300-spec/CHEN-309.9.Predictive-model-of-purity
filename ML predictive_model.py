

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


## Loading Excel file with dataset
dataset="Distillation Column Dataset.xlsx"

df=pd.read_excel (dataset)
#display basic info of data
df.info()

#display the first few rows
df.head()
## data preprocessing /cleaning
#check for missing values
df.isnull().sum()
df.describe()
#drop column with constant values
column_to_drop =['Sensor9', 'Sensor10', 'Sensor16']
df=df.drop(columns=column_to_drop,axis=1)
df.head()
#Rename columns for better readability
column_rename_map = {
    'Sensor1': 'Liquid%inCondensor',
    'Sensor2': 'Condenser_Pressure',
    'Sensor3': 'Liquid%inReboiler',
    'Sensor4': 'Mass_Flow_Rate_in_Feed_Flow',
    'Sensor5': 'Mass_Flow_Rate_in_Top_outlet_stream',
    'Sensor6': 'Net_Mass_Flow_in_main_tower',
    'Sensor7': 'Mole Fraction HX at reboiler',
    'Sensor8': 'HX Mole Fraction in Top Outler Stream',
    'Sensor11': 'Feed_Tray_Temp',
    'Sensor12': 'Main_Tower_Pressure',
    'Sensor13': 'Bottom Tower Pressure',
    'Sensor14': 'Top Tower Pressure',
    'Sensor15': 'Reflux_Ratio'
}
df.rename(columns=column_rename_map, inplace=True)



nits = {
    'Liquid%inCondensor': '%',
    'Condenser_Pressure': 'Pa',  # Assuming pressure in Pascals
    'Liquid%inReboiler': '%',
    'Mass_Flow_Rate_in_Feed_Flow': 'kg/s',
    'Mass_Flow_Rate_in_Top_outlet_stream': 'kg/s',
    'Net_Mass_Flow_in_main_tower': 'kg/s',
    'Feed_Tray_Temp': '°C',  # Assuming temperature in Celsius
    'Main_Tower_Pressure': 'Pa',  # Assuming pressure in Pascals
    'Mole Fraction HX at reboiler': '',
    'HX Mole Fraction in Top Outler Stream': '',
    'Bottom Tower Pressure': 'Pa',
    'Top Tower Pressure': 'Pa',
    'Reflux_Ratio': ''}


#view change
df.head()
# Define the sensors and target properties
sensors = ['Liquid%inCondensor', 'Condenser_Pressure', 'Liquid%inReboiler', 'Mass_Flow_Rate_in_Feed_Flow', 'Mass_Flow_Rate_in_Top_outlet_stream', 'Net_Mass_Flow_in_main_tower','Mole Fraction HX at reboiler','HX Mole Fraction in Top Outler Stream', 'Feed_Tray_Temp', 'Main_Tower_Pressure','Bottom Tower Pressure','Top Tower Pressure','Reflux_Ratio']
target_properties = [ 'MoleFractionHX','MoleFractionTX']
correlation_matrix = df[sensors + target_properties].corr()
plt.figure(figsize=(12, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,  # Display values in each cell
    cmap='coolwarm',
    fmt='.2f',square=True)

plt.title('Correlation Matrix Heatmap (Filtered Data)')
plt.show()
#DEFINING TARGET VARIABLE FOR TRAINING MY MODEL
X = df.drop(columns=['MoleFractionTX', 'MoleFractionHX'])
y_TX = df['MoleFractionTX']

#TRAIN TEST SPLIT

X_train, X_test, y_TX_train, y_TX_test= train_test_split(X, y_TX, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Scale the target variables
target_scaler = StandardScaler()
y_TX_train_scaled = target_scaler.fit_transform(y_TX_train.values.reshape(-1, 1))
y_TX_test_scaled = target_scaler.transform(y_TX_test.values.reshape(-1, 1))

# TRAINING THE MODEL FOR MOLE FRACTION IN TOP OUTLET STREAM (TX) WITH RANDOM FOREST REGRESSOR

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_TX_train)
y_TX_pred_rf = rf.predict(X_test_scaled)


# Checking prediction power with R^2 score
r2_rf = r2_score(y_TX_test, y_TX_pred_rf)

print(f"R^2 Score for Random Forest Regressor (TX): {r2_rf:.4f}")
#TEST AND VALIDATION USING MAE(ABSOLUTE VALUE ERROR)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_TX_test_scaled,y_TX_pred_rf)


print('mae=',mae)
#Checking how many features to be included as new coditions
print("Scaler expects these features:")
print(scaler.feature_names_in_)
print("Number of features:", len(scaler.feature_names_in_))

new_conditions = pd.DataFrame({
    'Liquid%inCondensor': [2022],
    'Condenser_Pressure': [101.7],
    'Liquid%inReboiler': [273.4],
    'Mass_Flow_Rate_in_Feed_Flow': [5000.7],
    'Mass_Flow_Rate_in_Top_outlet_stream': [2000],
    'Net_Mass_Flow_in_main_tower': [7000.3],
    'Mole Fraction HX at reboiler': [3924.8],
    'HX Mole Fraction in Top Outler Stream': [4500.5],  
    'Feed_Tray_Temp': [0.01],
    'Main_Tower_Pressure': [450.8],
    'Bottom Tower Pressure': [656.1],
    'Top Tower Pressure': [556.2],
    'Reflux_Ratio': [8.8],  
})

new_conditions_scaled = scaler.transform(new_conditions)
predicted_TX = rf.predict(new_conditions_scaled)
print("Predicted Mole Fraction TX:", predicted_TX[0])
print(f"product purity: {predicted_TX[0]*100:.2f}%")

# Mole fraction TX(top outlet distillate)was used as a yardstick for percentage purity measurement




## LINEAR REGRESSION MODEL

from sklearn.linear_model import LinearRegression

# Linear Regression for MoleFractionTX
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_TX_train)
y_TX_pred_lin = lin_reg.predict(X_test_scaled)
TX_r2_lin = r2_score(y_TX_test, y_TX_pred_lin)

print(f"Linear Regression Model for MoleFractionTX R^2: {TX_r2_lin}")

#TEST AND VALIDATION USING MAE(ABSOLUTE VALUE ERROR)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_TX_test_scaled,y_TX_pred_lin)


print('mae=',mae)
