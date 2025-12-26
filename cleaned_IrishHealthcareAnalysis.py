import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, classification_report, precision_score, recall_score

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

df_procedures = pd.read_csv('Procedures.csv')
df_waits = pd.read_csv('Waits.csv')

df_proc_clean = df_procedures.copy()

initial_rows = len(df_proc_clean)
missing_count = df_proc_clean['VALUE'].isnull().sum()

if missing_count > 0:
    df_proc_clean = df_proc_clean.dropna(subset=['VALUE'])
    rows_dropped = initial_rows - len(df_proc_clean)

for col in df_proc_clean.columns:
    col_type = df_proc_clean[col].dtype

    if col_type != object:
        c_min = df_proc_clean[col].min()
        c_max = df_proc_clean[col].max()

        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_proc_clean[col] = df_proc_clean[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_proc_clean[col] = df_proc_clean[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_proc_clean[col] = df_proc_clean[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df_proc_clean[col] = df_proc_clean[col].astype(np.float32)
            else:
                df_proc_clean[col] = df_proc_clean[col].astype(np.float32)

df_proc_clean = df_proc_clean.rename(columns={'VALUE': 'ProcedureCount'})

label_encoders = {}

le_sex = LabelEncoder()
df_proc_clean['SexEncoded'] = le_sex.fit_transform(df_proc_clean['Sex'].astype(str))
label_encoders['Sex'] = le_sex

le_procedure = LabelEncoder()
df_proc_clean['TypeofProcedureEncoded'] = le_procedure.fit_transform(
    df_proc_clean['Type of Procedure'].astype(str)
)
label_encoders['Type of Procedure'] = le_procedure

le_area = LabelEncoder()
df_proc_clean['AreaEncoded'] = le_area.fit_transform(df_proc_clean['Area'].astype(str))
label_encoders['Area'] = le_area

df_waits_clean = df_waits.copy()

df_waits_clean['Year'] = df_waits_clean['TLIST(M1)'].str[:4].astype(int)

df_waits_clean = df_waits_clean.rename(columns={'VALUE': 'WaitPercentage'})

df_waits_annual = df_waits_clean.groupby('Year').agg({
    'WaitPercentage': 'mean'
}).reset_index()
df_waits_annual.columns = ['Year', 'AvgWaitPercentage']

df_integrated = df_proc_clean.merge(
    df_waits_annual,
    on='Year',
    how='left'
)

missing_counts = df_integrated.isnull().sum()
missing_df = pd.DataFrame({
    'Column': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Missing_Percentage': (np.array(missing_counts.values) / float(len(df_integrated)) * 100)
})
missing_df = missing_df[missing_df['Missing_Count'] > 0]

if len(missing_df) > 0:
    cols_to_check = [c for c in df_integrated.columns if c != 'AvgWaitPercentage']
    initial_len = len(df_integrated)
    df_integrated = df_integrated.dropna(subset=cols_to_check)

    if df_integrated['AvgWaitPercentage'].isnull().sum() > 0:
        non_null_count = df_integrated['AvgWaitPercentage'].notna().sum()

        if non_null_count > 0:
            remaining_nan = df_integrated['AvgWaitPercentage'].isnull().sum()
            if remaining_nan > 0:
                global_median = df_integrated['AvgWaitPercentage'].median()
                df_integrated['AvgWaitPercentage'].fillna(global_median, inplace=True)

        else:
            def estimate_wait_percentage(row):
                proc_type = str(row['Type of Procedure']).lower()
                if 'emergency' in proc_type or 'urgent' in proc_type:
                    return 85.0
                elif 'diagnostic' in proc_type or 'imaging' in proc_type:
                    return 78.0
                elif 'surgery' in proc_type or 'invasive' in proc_type:
                    return 70.0
                else:
                    return 75.0

            df_integrated['AvgWaitPercentage'] = df_integrated.apply(
                estimate_wait_percentage, axis=1
            )

for col in df_integrated.columns:
    col_type = df_integrated[col].dtype

    if col_type != object:
        c_min = df_integrated[col].min()
        c_max = df_integrated[col].max()

        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_integrated[col] = df_integrated[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_integrated[col] = df_integrated[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_integrated[col] = df_integrated[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df_integrated[col] = df_integrated[col].astype(np.float32)
            else:
                df_integrated[col] = df_integrated[col].astype(np.float32)

Q1 = df_integrated['ProcedureCount'].quantile(0.25)
Q3 = df_integrated['ProcedureCount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_integrated['ProcedureCount_Original'] = df_integrated['ProcedureCount'].copy()
df_integrated['ProcedureCount'] = df_integrated['ProcedureCount'].clip(
    lower=lower_bound, upper=upper_bound
)

df_integrated['CurrentBottleneckScore'] = (
    df_integrated['ProcedureCount'] * (100 - df_integrated['AvgWaitPercentage'])
)

df_integrated = df_integrated.sort_values(['Type of Procedure', 'Area', 'Sex', 'Year'])

df_integrated['NextYearBottleneckScore'] = df_integrated.groupby(
    ['Type of Procedure', 'Area', 'Sex']
)['CurrentBottleneckScore'].shift(-1)

SPLIT_YEAR = 2017
train_mask = df_integrated['Year'] <= SPLIT_YEAR
bottleneck_threshold = df_integrated.loc[train_mask, 'CurrentBottleneckScore'].quantile(0.75)

df_integrated['FutureHighBottleneck'] = (
    df_integrated['NextYearBottleneckScore'] > bottleneck_threshold
).astype(float)

df_integrated['PrevYearProcedures'] = df_integrated.groupby(
    ['Type of Procedure', 'Area', 'Sex']
)['ProcedureCount'].shift(1)

df_integrated['PrevYearWaitPct'] = df_integrated.groupby(
    ['Type of Procedure', 'Area', 'Sex']
)['AvgWaitPercentage'].shift(1)

df_integrated['Growth2Year'] = df_integrated.groupby(
    ['Type of Procedure', 'Area', 'Sex']
)['ProcedureCount'].pct_change(periods=2).shift(1)
df_integrated['Growth2Year'] = df_integrated['Growth2Year'].replace([np.inf, -np.inf], [10.0, -1.0])
df_integrated['Growth2Year'] = df_integrated['Growth2Year'].fillna(0)

df_integrated['PrevYearBottleneckScore'] = df_integrated.groupby(
    ['Type of Procedure', 'Area', 'Sex']
)['CurrentBottleneckScore'].shift(1)
df_integrated['PrevYearBottleneckScore'] = df_integrated['PrevYearBottleneckScore'].fillna(0)

df_integrated['Avg3YearProcedures'] = (
    df_integrated.groupby(['Type of Procedure', 'Area', 'Sex'])['ProcedureCount']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

df_integrated['Volatility3Year'] = (
    df_integrated.groupby(['Type of Procedure', 'Area', 'Sex'])['ProcedureCount']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).std())
)
df_integrated['Volatility3Year'] = df_integrated['Volatility3Year'].fillna(0)

area_yearly_totals = df_integrated.groupby(['Area', 'Year'])['ProcedureCount'].sum().reset_index()
area_yearly_totals = area_yearly_totals.sort_values(['Area', 'Year'])

area_yearly_totals['AreaCumulativeProcedures'] = (
    area_yearly_totals.groupby('Area')['ProcedureCount']
    .cumsum()
    .shift(1)
)

df_integrated = df_integrated.drop(columns=['AreaCumulativeProcedures'], errors='ignore')
df_integrated = pd.merge(
    df_integrated,
    area_yearly_totals[['Area', 'Year', 'AreaCumulativeProcedures']],
    on=['Area', 'Year'],
    how='left'
)
df_integrated['AreaCumulativeProcedures'] = df_integrated['AreaCumulativeProcedures'].fillna(0)

df_integrated['LogVolume'] = np.log1p(df_integrated['PrevYearProcedures']).round(1)

df_integrated['RoundedWait'] = df_integrated['PrevYearWaitPct'].round(0)

df_integrated['RoundedGrowth'] = df_integrated['Growth2Year'].round(1)

df_integrated['RoundedVolatility'] = df_integrated['Volatility3Year'].round(1)

os.makedirs('OP1', exist_ok=True)
df_integrated.to_csv('OP1/IntegratedHealthcareData.csv', index=False)

os.makedirs('OP1/visualizations', exist_ok=True)

temporal = df_integrated.groupby('Year').agg({
    'ProcedureCount': 'sum',
    'AvgWaitPercentage': 'mean',
    'CurrentBottleneckScore': 'mean'
}).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax1 = axes[0]
ax1_twin = ax1.twinx()

ax1.plot(temporal['Year'], temporal['ProcedureCount'],
         'o-', color='#2E86AB', linewidth=2.5, markersize=8)
ax1.set_ylabel('Total Procedure Count', fontsize=12, fontweight='bold', color='#2E86AB')
ax1.tick_params(axis='y', labelcolor='#2E86AB')

ax1_twin.plot(temporal['Year'], temporal['AvgWaitPercentage'],
              's-', color='#A23B72', linewidth=2.5, markersize=8)
ax1_twin.set_ylabel('Avg Wait %', fontsize=12, fontweight='bold', color='#A23B72')
ax1_twin.tick_params(axis='y', labelcolor='#A23B72')

ax1.set_title('Healthcare Demand vs Service Delivery Over Time',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(temporal['Year'], temporal['CurrentBottleneckScore'],
         'D-', color='#F18F01', linewidth=2.5, markersize=8)
ax2.fill_between(temporal['Year'], temporal['CurrentBottleneckScore'], alpha=0.3, color='#F18F01')
ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Bottleneck Score', fontsize=12, fontweight='bold')
ax2.set_title('Bottleneck Severity Trend', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('OP1/visualizations/temporal_trends.png', dpi=300, bbox_inches='tight')
plt.close()

top_bottlenecks = df_integrated.groupby('Type of Procedure').agg({
    'CurrentBottleneckScore': 'mean',
    'ProcedureCount': 'mean',
    'AvgWaitPercentage': 'mean'
}).sort_values('CurrentBottleneckScore', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(14, 8))
procedures = top_bottlenecks.index
scores = top_bottlenecks['CurrentBottleneckScore']
colors = plt.cm.get_cmap('RdYlGn_r')(np.linspace(0.2, 0.8, len(procedures)))

bars = ax.barh(range(len(procedures)), scores, color=colors)
ax.set_yticks(range(len(procedures)))
ax.set_yticklabels(procedures, fontsize=10)
ax.set_xlabel('Average Bottleneck Score', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Healthcare Bottleneck Procedures', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.text(score, i, f' {score:,.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('OP1/visualizations/top_bottlenecks.png', dpi=300, bbox_inches='tight')
plt.close()

geo_analysis = df_integrated.groupby('Area').agg({
    'CurrentBottleneckScore': 'mean',
    'ProcedureCount': 'sum',
    'AvgWaitPercentage': 'mean'
}).sort_values('CurrentBottleneckScore', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(14, 8))
scatter = ax.scatter(
    geo_analysis['ProcedureCount'],
    geo_analysis['AvgWaitPercentage'],
    s=geo_analysis['CurrentBottleneckScore'] / 100,
    c=geo_analysis['CurrentBottleneckScore'],
    cmap='viridis',
    alpha=0.6,
    edgecolors='black',
    linewidth=1.5
)

ax.set_xlabel('Total Procedure Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Avg Wait Percentage', fontsize=12, fontweight='bold')
ax.set_title('Geographic Healthcare Analysis\n(Size & Color=Bottleneck Score)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.colorbar(scatter, ax=ax, label='Bottleneck Score')
plt.tight_layout()
plt.savefig('OP1/visualizations/geographic_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

corr_cols = ['Year', 'PrevYearProcedures', 'PrevYearWaitPct',
             'Growth2Year', 'Avg3YearProcedures', 'Volatility3Year']
df_corr = df_integrated[[c for c in corr_cols if c in df_integrated.columns]].dropna()
correlation_matrix = df_corr.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('OP1/visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

feature_cols = [
    'SexEncoded',
    'LogVolume',
    'RoundedWait',
    'RoundedGrowth',
    'RoundedVolatility'
]

df_integrated['IsLastYear'] = df_integrated.groupby(
    ['Type of Procedure', 'Area', 'Sex']
)['Year'].transform(lambda x: x == x.max())

rows_before = len(df_integrated)
df_ml = df_integrated[~df_integrated['IsLastYear']].copy()
rows_after = len(df_ml)

df_ml = df_ml.dropna(subset=feature_cols)

df_ml = df_ml.dropna(subset=['FutureHighBottleneck'])

for col in df_ml.columns:
    col_type = df_ml[col].dtype

    if col_type != object:
        c_min = df_ml[col].min()
        c_max = df_ml[col].max()

        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_ml[col] = df_ml[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_ml[col] = df_ml[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_ml[col] = df_ml[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df_ml[col] = df_ml[col].astype(np.float32)
            else:
                df_ml[col] = df_ml[col].astype(np.float32)

X = df_ml[feature_cols].copy()
y_classification = df_ml['FutureHighBottleneck'].copy()
y_regression = df_ml['NextYearBottleneckScore'].copy()

train_mask = df_ml['Year'] <= SPLIT_YEAR
test_mask = df_ml['Year'] > SPLIT_YEAR

X_train = X[train_mask].copy()
X_test = X[test_mask].copy()

y_train_clf = y_classification[train_mask].copy()
y_test_clf = y_classification[test_mask].copy()

y_train_reg = y_regression[train_mask].copy()
y_test_reg = y_regression[test_mask].copy()

X_train_clf, X_test_clf = X_train, X_test
X_train_reg, X_test_reg = X_train, X_test

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

X_train_clf_balanced = X_train_clf_scaled
y_train_clf_balanced = y_train_clf

classification_results = []

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    C=0.1
)
lr_model.fit(X_train_clf_balanced, y_train_clf_balanced)
y_pred_lr = lr_model.predict(X_test_clf_scaled)
acc_lr = accuracy_score(y_test_clf, y_pred_lr)
f1_lr = f1_score(y_test_clf, y_pred_lr, zero_division=0)
prec_lr = precision_score(y_test_clf, y_pred_lr, zero_division=0)
rec_lr = recall_score(y_test_clf, y_pred_lr, zero_division=0)
classification_results.append({
    'Model': 'Logistic Regression',
    'Accuracy': acc_lr,
    'Precision': prec_lr,
    'Recall': rec_lr,
    'F1-Score': f1_lr
})

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train_clf_balanced, y_train_clf_balanced)
y_pred_rf = rf_clf.predict(X_test_clf_scaled)
acc_rf = accuracy_score(y_test_clf, y_pred_rf)
f1_rf = f1_score(y_test_clf, y_pred_rf, zero_division=0)
prec_rf = precision_score(y_test_clf, y_pred_rf, zero_division=0)
rec_rf = recall_score(y_test_clf, y_pred_rf, zero_division=0)
classification_results.append({
    'Model': 'Random Forest',
    'Accuracy': acc_rf,
    'Precision': prec_rf,
    'Recall': rec_rf,
    'F1-Score': f1_rf
})

gb_clf = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=7,
    learning_rate=0.05,
    random_state=42
)
gb_clf.fit(X_train_clf_balanced, y_train_clf_balanced)
y_pred_gb = gb_clf.predict(X_test_clf_scaled)
acc_gb = accuracy_score(y_test_clf, y_pred_gb)
f1_gb = f1_score(y_test_clf, y_pred_gb, zero_division=0)
prec_gb = precision_score(y_test_clf, y_pred_gb, zero_division=0)
rec_gb = recall_score(y_test_clf, y_pred_gb, zero_division=0)
classification_results.append({
    'Model': 'HistGradientBoosting',
    'Accuracy': acc_gb,
    'Precision': prec_gb,
    'Recall': rec_gb,
    'F1-Score': f1_gb
})

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'max_iter': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'l2_regularization': [0, 0.1, 1.0]
}

gb_base = HistGradientBoostingClassifier(random_state=42)

auto_clf = RandomizedSearchCV(
    estimator=gb_base,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

auto_clf.fit(X_train_clf_balanced, y_train_clf_balanced)

y_pred_auto = auto_clf.predict(X_test_clf_scaled)

acc_auto = accuracy_score(y_test_clf, y_pred_auto)
f1_auto = f1_score(y_test_clf, y_pred_auto, zero_division=0)
prec_auto = precision_score(y_test_clf, y_pred_auto, zero_division=0)
rec_auto = recall_score(y_test_clf, y_pred_auto, zero_division=0)

classification_results.append({
    'Model': 'AutoML (Tuned HistGBM)',
    'Accuracy': acc_auto,
    'Precision': prec_auto,
    'Recall': rec_auto,
    'F1-Score': f1_auto
})

clf_results_df = pd.DataFrame(classification_results)

clf_results_df.to_csv('OP1/classification_results.csv', index=False)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

regression_results = []

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_ridge = ridge_model.predict(X_test_reg_scaled)
mse_ridge = mean_squared_error(y_test_reg, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test_reg, y_pred_ridge)
r2_ridge = r2_score(y_test_reg, y_pred_ridge)
regression_results.append({
    'Model': 'Ridge Regression',
    'RMSE': np.sqrt(mse_ridge),
    'MAE': mae_ridge,
    'R²': r2_ridge
})

lasso_model = Lasso(alpha=1.0, random_state=42)
lasso_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_lasso = lasso_model.predict(X_test_reg_scaled)
mse_lasso = mean_squared_error(y_test_reg, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test_reg, y_pred_lasso)
r2_lasso = r2_score(y_test_reg, y_pred_lasso)
regression_results.append({
    'Model': 'Lasso Regression',
    'RMSE': np.sqrt(mse_lasso),
    'MAE': mae_lasso,
    'R²': r2_lasso
})

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test_reg_scaled)
mse_rf_reg = mean_squared_error(y_test_reg, y_pred_rf_reg)
mae_rf_reg = mean_absolute_error(y_test_reg, y_pred_rf_reg)
r2_rf_reg = r2_score(y_test_reg, y_pred_rf_reg)
regression_results.append({
    'Model': 'Random Forest',
    'RMSE': np.sqrt(mse_rf_reg),
    'MAE': mae_rf_reg,
    'R²': r2_rf_reg
})

gb_reg = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=5,
    random_state=42
)
gb_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_gb_reg = gb_reg.predict(X_test_reg_scaled)
mse_gb_reg = mean_squared_error(y_test_reg, y_pred_gb_reg)
mae_gb_reg = mean_absolute_error(y_test_reg, y_pred_gb_reg)
r2_gb_reg = r2_score(y_test_reg, y_pred_gb_reg)
regression_results.append({
    'Model': 'HistGradientBoosting',
    'RMSE': np.sqrt(mse_gb_reg),
    'MAE': mae_gb_reg,
    'R²': r2_gb_reg
})

param_dist_reg = {
    'max_iter': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'l2_regularization': [0, 0.1, 1.0]
}

gb_base_reg = HistGradientBoostingRegressor(random_state=42)

auto_reg = RandomizedSearchCV(
    estimator=gb_base_reg,
    param_distributions=param_dist_reg,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

auto_reg.fit(X_train_reg_scaled, y_train_reg)

y_pred_auto_reg = auto_reg.predict(X_test_reg_scaled)

mse_auto_reg = mean_squared_error(y_test_reg, y_pred_auto_reg)
mae_auto_reg = mean_absolute_error(y_test_reg, y_pred_auto_reg)
r2_auto_reg = r2_score(y_test_reg, y_pred_auto_reg)

regression_results.append({
    'Model': 'AutoML (Tuned HistGBM)',
    'RMSE': np.sqrt(mse_auto_reg),
    'MAE': mae_auto_reg,
    'R²': r2_auto_reg
})

reg_results_df = pd.DataFrame(regression_results)

reg_results_df.to_csv('OP1/regression_results.csv', index=False)

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_clf.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.get_cmap('viridis')(np.linspace(0.3, 0.9, len(feature_importance)))
bars = ax.barh(range(len(feature_importance)), feature_importance['Importance'], color=colors)

ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['Feature'], fontsize=11)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance for Bottleneck Prediction', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, feature_importance['Importance'])):
    ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('OP1/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

feature_importance.to_csv('OP1/feature_importance.csv', index=False)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(rf_clf, X_train_clf_scaled, y_train_clf,
                            cv=skf, scoring='accuracy')

fig, ax = plt.subplots(figsize=(12, 6))
folds = [f'Fold {i+1}' for i in range(10)]
colors = ['#06A77D' if score >= cv_scores.mean() else '#D62828' for score in cv_scores]

bars = ax.bar(folds, cv_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(cv_scores.mean(), color='blue', linestyle='--', linewidth=2,
           label=f'Mean: {cv_scores.mean():.4f}')
ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
ax.set_title('10-Fold Cross-Validation Results\n(Green=Above Mean, Red=Below Mean)',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar, score in zip(bars, cv_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('OP1/visualizations/cross_validation_results.png', dpi=300, bbox_inches='tight')
plt.close()

cv_results_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(10)],
    'Accuracy': cv_scores
})
cv_results_df.to_csv('OP1/cv_results.csv', index=False)

best_model_idx = clf_results_df['F1-Score'].idxmax()
best_model_name = clf_results_df.loc[best_model_idx, 'Model']

if best_model_name == 'Random Forest':
    best_clf_model = rf_clf
elif best_model_name == 'HistGradientBoosting':
    best_clf_model = gb_clf
elif best_model_name == 'AutoML (Tuned HistGBM)':
    best_clf_model = auto_clf
else:
    best_clf_model = lr_model

best_reg_idx = reg_results_df['R²'].idxmax()
best_reg_name = reg_results_df.loc[best_reg_idx, 'Model']

if best_reg_name == 'Random Forest':
    best_reg_model = rf_reg
elif best_reg_name == 'HistGradientBoosting':
    best_reg_model = gb_reg
elif best_reg_name == 'AutoML (Tuned HistGBM)':
    best_reg_model = auto_reg
elif best_reg_name == 'Lasso Regression':
    best_reg_model = lasso_model
else:
    best_reg_model = ridge_model

y_pred_bottleneck = best_clf_model.predict(X_test_clf_scaled)
y_pred_score = best_reg_model.predict(X_test_reg_scaled)

predictions_df = pd.DataFrame({
    'Year': df_ml.loc[test_mask, 'Year'].values,
    'Area': df_ml.loc[test_mask, 'Area'].values,
    'Procedure': df_ml.loc[test_mask, 'Type of Procedure'].values,
    'Sex': df_ml.loc[test_mask, 'Sex'].values,
    'Predicted_High_Bottleneck': y_pred_bottleneck,
    'Predicted_Bottleneck_Score': y_pred_score,
    'Actual_High_Bottleneck': y_test_clf.values,
    'Actual_Bottleneck_Score': y_test_reg.values
})

predictions_df['Prediction_Correct'] = (
    predictions_df['Predicted_High_Bottleneck'] == predictions_df['Actual_High_Bottleneck']
).astype(int)

high_risk_predictions = predictions_df[
    predictions_df['Predicted_High_Bottleneck'] == 1.0
].copy()

high_risk_predictions = high_risk_predictions.sort_values(
    'Predicted_Bottleneck_Score', ascending=False
)

predictions_df.to_csv('OP1/future_bottleneck_predictions.csv', index=False)

high_risk_predictions.to_csv('OP1/high_risk_areas_for_allocation.csv', index=False)

area_summary = high_risk_predictions.groupby('Area').agg({
    'Predicted_Bottleneck_Score': ['count', 'mean', 'max'],
    'Prediction_Correct': 'mean'
}).round(2)
area_summary.columns = ['High_Risk_Count', 'Avg_Predicted_Score', 'Max_Predicted_Score', 'Accuracy']
area_summary = area_summary.sort_values('High_Risk_Count', ascending=False)

area_summary.to_csv('OP1/area_level_resource_priorities.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

top_areas = high_risk_predictions['Area'].value_counts().head(15)
colors = plt.cm.get_cmap('Reds')(np.linspace(0.4, 0.9, len(top_areas)))
axes[0].barh(range(len(top_areas)), top_areas.values, color=colors)
axes[0].set_yticks(range(len(top_areas)))
axes[0].set_yticklabels(top_areas.index, fontsize=9)
axes[0].set_xlabel('Number of Predicted High-Risk Cases', fontsize=11, fontweight='bold')
axes[0].set_title('Top 15 Areas with Predicted Bottlenecks\n(2018-2019)', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

year_accuracy = predictions_df.groupby('Year')['Prediction_Correct'].mean() * 100
axes[1].bar(year_accuracy.index.astype(str), year_accuracy.values,
           color=['#06A77D', '#F4A261'], edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Prediction Accuracy (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Model Prediction Accuracy by Year\n(Future Test Set)', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 100)
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(year_accuracy.values):
    axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('OP1/visualizations/resource_allocation_predictions.png', dpi=300, bbox_inches='tight')
plt.close()