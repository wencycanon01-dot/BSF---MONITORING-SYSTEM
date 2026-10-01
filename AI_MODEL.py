import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

file_name = "THESIS BSF DATASET.xlsx"

print("\n" + "="*60)
print("     BSF LARVAE YIELD PREDICTION & COMPARATIVE ANALYSIS")
print("="*60)

if not os.path.exists(file_name):
    print(f"\n[ERROR] Cannot find '{file_name}'.")
    print("Ensure the Python file and the Excel dataset are in the same directory.")
    exit()

print("\n[1] LOADING EXCEL DATASET...")
df = pd.read_excel(file_name)
print(f"SUCCESS: Dataset loaded ({len(df)} rows).")

y = df['Yield_g']
X = df[['Day', 'Substrate', 'Cumulative_Feed_Given (g)', 'Temperature_C', 'Humidity_Percent']]

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['Day']))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Day', 'Cumulative_Feed_Given (g)', 'Temperature_C', 'Humidity_Percent']),
        ('cat', OneHotEncoder(drop='first'), ['Substrate'])
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=4, random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=5, weights='distance')
}

print("\n[2] INITIATING TRAINING AND TESTING PHASE...\n")
print("="*60)

best_r2 = -float('inf')
best_model_name = ""

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    test_preds = model.predict(X_test_processed)
    
    r2 = r2_score(y_test, test_preds)
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    mape = mean_absolute_percentage_error(y_test, test_preds)
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
    
    print(f"MODEL: {name.upper()}")

    print(f"  R-Squared (R² Score): {r2 * 100:.2f}%")
    print(f"  RMSE (Root Error):    {rmse:.2f} grams")
    print(f"  MAE (Average Error):  +/- {mae:.2f} grams")
    print(f"  MAPE (Percent Error): {mape * 100:.2f}%")
    
    if r2 > 0.90:
        print("  [STATUS] Strong predictive capability on unseen data.")
    elif r2 > 0.80:
        print("  [STATUS] Acceptable generalization.")
    else:
        print("  [STATUS] Poor fit. Needs tuning.")
        
    print("-" * 60)

print("\n[3] COMPARATIVE ANALYSIS COMPLETE!")
print(f"Based on the R² Score and MAE/RMSE, {best_model_name} performed best among the evaluated models on the specified test set.")
print("="*60)

print("\n[4] EXPORTING ALL MODELS FOR DASHBOARD INTEGRATION...")

joblib.dump(models["Decision Tree"], 'bsf_dt_model.pkl')
joblib.dump(models["Random Forest"], 'bsf_rf_model.pkl')
joblib.dump(models["K-Nearest Neighbors (KNN)"], 'bsf_knn_model.pkl')
joblib.dump(preprocessor, 'bsf_preprocessor.pkl')

print("SUCCESS: All models and preprocessor saved successfully.")
print("The AI is ready for web deployment!\n")