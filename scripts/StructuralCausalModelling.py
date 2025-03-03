########## Least Squares with interventions ##########
# Author: Myrthe Leijnse

### Imports ###
import os
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import shapiro
from scipy.stats import t
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

### Directory ###
os.getcwd()

### Reading Data ###
# Preprocessed
file_path = "data/Input_JPCMCI/REVIEW_Quantiles_LWEGAPS_ERA5_EVI_POP_discharge_InterpolatedNoise_monmean_detrend_2002-2019_allhotspots.csv"
df_allhotspots = pd.read_csv(file_path)
hotspotnames = df_allhotspots['hotspot'].unique().tolist()

# Original
lwe_df = pd.read_csv("/eejit/home/5738091/data/Causality/monthly_csv/absolute_values/LWE_2002-2023_ZonalInterpolatedGaps_hotspots_monthly.csv")
pr_df = pd.read_csv("/eejit/home/5738091/data/Causality/monthly_csv/absolute_values/ERA5pr_1950-2023_Zonalmean_summed_hotspots.csv")
t2m_df = pd.read_csv("/eejit/home/5738091/data/Causality/monthly_csv/absolute_values/ERA5t2m_1950-2022_Zonalmean_hotspots.csv")
pop_df = pd.read_csv("/eejit/home/5738091/data/Causality/monthly_csv/pop_2000-2019!_ZonalInterpolated_hotspots_monthly.csv")
EVI_df = pd.read_csv("/eejit/home/5738091/data/Causality/monthly_csv/absolute_values/MODIS_EVI_irrarea_2001_2022_Zonalmmean_hotspots.csv")
Q_df = pd.read_csv("/eejit/home/5738091/data/Causality/monthly_csv/absolute_values/discharge_1980-2019_Zonalmean_hotspots.csv")
dataframes = [lwe_df, pr_df, t2m_df, pop_df, EVI_df, Q_df]

df = dataframes[0]
for frame in dataframes[1:]:
    df = pd.merge(df, frame, on = ["time", "hotspot"], how = "inner")
df_intv = df
df_intv.reset_index(drop=True, inplace=True)

### Functions ###
def plot_variable(df, df_new, hotspot_name, variable):
    # Filter the DataFrame for the specified hotspot
    df_test = df[df["hotspot"] == hotspot_name]
    df_test2 = df_new[df_new["hotspot"] == hotspot_name]
    
    # Plot the "population" column
    plt.figure(figsize=(10, 6))
    plt.plot(df_test["time"], df_test[variable], marker='o', linestyle='-')
    plt.plot(df_test2["time"], df_test2[variable], marker='o', linestyle='-', color='orange')
    plt.title(f'{variable} over Time for {hotspot_name}')
    plt.xlabel('Time')
    plt.ylabel(f'{variable}')
    plt.xticks(df_test['time'][::6], rotation=45, ha="right")
    plt.legend(["Original", "Detrended"])
    plt.grid(True)
    plt.show() 

# Preprocess data functions
def monavg_df(df, variable, variable_intv):
    df_new = pd.DataFrame()
    df_original = df[["hotspot", "time", variable]]
    df_intv = df[["hotspot", "time", variable_intv]]
    for hotspot in df["hotspot"].unique():
        df_shp = df_intv[df_intv["hotspot"] == hotspot]
        df_shp_original = df_original[df_original["hotspot"] == hotspot]
        df_shp = df_shp.set_index(pd.to_datetime(df_shp["time"]))
        df_shp_original = df_shp_original.set_index(pd.to_datetime(df_shp_original["time"]))
        # Remove seasonal trend
        monthly_mean = df_shp_original.groupby(df_shp_original.index.month)[variable].transform('mean')
        df_shp = pd.DataFrame(df_shp[variable_intv]-monthly_mean)
        df_shp = df_shp.reset_index()
        df_shp["time"] = df_shp["time"].dt.strftime('%B %Y')
        df_shp["hotspot"] = hotspot
        df_shp = df_shp.rename(columns={0: variable_intv})
        df_new = pd.concat([df_new, df_shp])
    return(df_new) 

def normalize_data(df, df_original, variable, variable_intv):
    quantile_transformer = (QuantileTransformer(output_distribution='uniform'))
    hotspot_names = df_original["hotspot"].unique()
    dfs = []
    for hotspot in hotspot_names:
        df_input = df_original[df_original["hotspot"] == hotspot]
        df_new = df[df["hotspot"] == hotspot]
        normalized = quantile_transformer.fit_transform(df_input[variable].values.reshape(-1, 1))
        df_new_normalized = quantile_transformer.transform(df_new[variable_intv].values.reshape(-1, 1))
        df_new[variable_intv] = df_new_normalized
        dfs.append(df_new)
    result_df = pd.concat(dfs, ignore_index=True)
    return(result_df)

def detrend(df, variable):
    dfs = []
    df_original = df[["hotspot", "time", variable]]
    for hotspot in df["hotspot"].unique():
        df_shp_original = df_original[df_original["hotspot"] == hotspot]
        # Remember the timesteps with NaN values
        nan_indices = df_shp_original[df_shp_original[variable].isna()].index
        # Interpolate NaN values
        df_shp_original[variable] = df_shp_original[variable].interpolate(method='linear')
        X = np.arange(len(df_shp_original)).reshape(-1, 1)
        y = df_shp_original[variable].values
        model = LinearRegression().fit(X, y)
        trend = model.predict(X)
        df_shp_original[variable] -= trend
        df_shp_original.loc[nan_indices, variable] = np.nan
        dfs.append(df_shp_original)
    result_df = pd.concat(dfs, ignore_index=True)
    return(result_df)

def detrend_intv(df, df_original, variable, variable_intv):
    dfs = []
    df_original = df_original[["hotspot", "time", variable]]
    df_intv = df[["hotspot", "time", variable_intv]]
    for hotspot in df["hotspot"].unique():
        df_shp = df_intv[df_intv["hotspot"] == hotspot]
        df_shp_original = df_original[df_original["hotspot"] == hotspot]
        # Drop NaNs, detrend, and reinsert NaNs
        nan_indices = df_shp_original[df_shp_original[variable].isna()].index
        df_shp_original[variable] = df_shp_original[variable].interpolate(method='linear')
        # Calculate the trend using linear regression
        X = np.arange(len(df_shp_original)).reshape(-1, 1)
        y = df_shp_original[variable].values
        model = LinearRegression().fit(X, y)
        trend = model.predict(X)
        df_shp[variable_intv] -= trend
        df_shp.loc[nan_indices, variable_intv] = np.nan
        dfs.append(df_shp)
    result_df = pd.concat(dfs, ignore_index=True)
    return(result_df)

# NLS model functions
def func(alpha, t2m_1, tp):
    t2m_predicted = alpha[0] * t2m_1 + alpha[1] * tp + alpha[2]
    return t2m_predicted

def func6(alpha, evi_1, t2m_1, tp, pop, tp_1):
    evi_predicted = alpha[9] * evi_1 + alpha[10] * func(alpha, t2m_1, tp) + alpha[11] * pop + alpha[12] * tp + alpha[20] * tp_1 + alpha[13]
    return evi_predicted

def func7(alpha, Q_1, tp, t2m_1, pop):
    Q_predicted = alpha[15] * Q_1 + alpha[16] * tp + alpha[17] * func(alpha, t2m_1, tp) + alpha[21] * pop + alpha[18]
    return Q_predicted

def func2(alpha, t2m_1, tp, tp_1, pop, evi_1, lwe_1, lwe, Q_1):
    lwe_predicted = alpha[3] * func(alpha, t2m_1, tp) + alpha[4] * tp + alpha[5] * tp_1 + alpha[6] * pop + alpha[7] * func6(alpha, evi_1, t2m_1, tp, pop, tp_1) + alpha[8] + alpha[14] * lwe_1 + alpha[19] * func7(alpha, Q_1, tp, tp_1, pop)
    residual = lwe_predicted - lwe
    return residual

def func3(alpha, t2m_1, tp, tp_1, pop, evi_1, lwe_1, Q_1):
    lwe_predicted = alpha[3] * func(alpha, t2m_1, tp) + alpha[4] * tp + alpha[5] * tp_1 + alpha[6] * pop + alpha[7] * func6(alpha, evi_1, t2m_1, tp, pop, tp_1) + alpha[8] + alpha[14] * lwe_1 + alpha[19] * func7(alpha, Q_1, tp, tp_1, pop)
    return lwe_predicted

def func4(alpha, t2m_1, tp, tp_1, pop, evi_1, lwe_predicted, Q_1, i):
    lwe_predicted = alpha[3] * func(alpha, t2m_1[i-1:i], tp[i-1:i]) + alpha[4] * tp[i-1:i] + alpha[5] * tp_1[i-1:i] + alpha[6] * pop[i-1:i] + alpha[7] * func6(alpha, evi_1[i-1:i], t2m_1[i-1:i], tp[i-1:i], pop[i-1:i], tp_1[i-1:i]) + alpha[8] + alpha[14] * lwe_predicted + alpha[19] * func7(alpha, Q_1[i-1:i], tp[i-1:i], tp_1[i-1:i], pop[i-1:i])
    return np.array(lwe_predicted)

# NLS bootstrap function
def NLS(X_train, y_train):
    t2m_1 = np.array(X_train[:,0], dtype=float)
    tp = np.array(X_train[:,1], dtype=float)
    tp_1 = np.array(X_train[:,2], dtype=float)
    pop = np.array(X_train[:,3], dtype=float)
    evi_1 = np.array(X_train[:,4], dtype=float)
    lwe_1 = np.array(X_train[:,5], dtype=float)
    lwe = np.array(y_train, dtype=float)
    Q_1 = np.array(X_train[:,6], dtype=float)
    
    initial = np.array([0.5] * 22)
    fit = least_squares(func2, initial, method='lm', args=(t2m_1, tp, tp_1, pop, evi_1, lwe_1, lwe, Q_1))
    return fit.x, fit

def X_to_vars(X):
    t2m_full = np.array(X[:,0], dtype=float)
    tp_full = np.array(X[:,1], dtype=float)
    tp_1_full = np.array(X[:,2], dtype=float)
    pop_full = np.array(X[:,3], dtype=float)
    evi_1_full = np.array(X[:,4], dtype=float)
    lwe_1_full = np.array(X[:,5], dtype=float)
    Q_1_full = np.array(X[:,6], dtype=float)
    return t2m_full, tp_full, tp_1_full, pop_full, evi_1_full, lwe_1_full, Q_1_full

### Execution ###
# Interventions
intervention = "-10%"
if intervention == "-10%":
    degrees = -1
    percentage = 0.9
elif intervention == "-5%":
    degrees = -0.5
    percentage = 0.95
elif intervention == "+5%":
    degrees = 0.5
    percentage = 1.05
elif intervention == "+10%":
    degrees = 1
    percentage = 1.1
intervention_t2m = f"{degrees}K"
df_intv["t2m_intv"] = df_intv["t2m"] + degrees
df_intv["tp_intv"] = df_intv["tp"] * percentage
df_intv["Population_intv"] = df_intv["Population"] * percentage
df_intv["EVI_intv"] = df_intv["EVI"] * percentage
df_intv["lwe_thickness_intv"] = df_intv["lwe_thickness"] * percentage
df_intv["discharge_intv"] = df_intv["discharge"] * percentage

# Preprocess intervention data
variable_list = ["t2m", "tp", "EVI", "lwe_thickness", "discharge"]
variable_intv_list = ["t2m_intv", "tp_intv", "EVI_intv", "lwe_thickness_intv", "discharge_intv"]
df_qnorm = pd.DataFrame()
for variable, variable_intv in zip(variable_list, variable_intv_list):
    print(variable, variable_intv)
    df_variable_monavg = monavg_df(df_intv, variable, variable)
    df_variable_intv_monavg = monavg_df(df_intv, variable, variable_intv)
    df_variable_detrend = detrend(df_variable_monavg, variable)
    df_variable_intv_detrend = detrend_intv(df_variable_intv_monavg, df_variable_monavg, variable, variable_intv)
    df_variable = normalize_data(df_variable_monavg, df_variable_monavg, variable, variable)
    df_variable_intv = normalize_data(df_variable_intv_monavg, df_variable_monavg, variable, variable_intv)
    df_qnorm = pd.concat([df_qnorm, df_variable[variable], df_variable_intv[variable_intv]], axis=1)

pop = df_intv[["time","hotspot","Population", "Population_intv"]]
df_pop = normalize_data(pop, pop, "Population", "Population")
df_pop_intv = normalize_data(pop, pop, "Population", "Population_intv")
df_qnorm = pd.concat([df_intv["hotspot"], df_intv["time"], df_qnorm, df_pop["Population"], df_pop_intv["Population_intv"]], axis=1)

# Scenario selection
scenario1 = "No intervention"
scenario2 = f"{intervention} population"
scenario3 = f"{intervention_t2m} temperature"
scenario4 = f"{intervention} precipitation"
scenario5 = f"{intervention} EVI"
scenario6 = f"{intervention} LWE thickness"
scenario7 = f"{intervention} discharge"

df_scenario1 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI", "lwe_thickness", "Population", "discharge"]]
df_scenario2 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI", "lwe_thickness", "Population_intv", "discharge"]]
df_scenario3 = df_qnorm[["hotspot", "time", "t2m_intv", "tp", "EVI", "lwe_thickness", "Population", "discharge"]]
df_scenario4 = df_qnorm[["hotspot", "time", "t2m", "tp_intv", "EVI", "lwe_thickness", "Population", "discharge"]]
df_scenario5 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI_intv", "lwe_thickness", "Population", "discharge"]]
df_scenario6 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI", "lwe_thickness_intv", "Population", "discharge"]]
df_scenario7 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI", "lwe_thickness", "Population", "discharge_intv"]]

# write csv of scenario1
df_scenario1.to_csv("data/Input_JPCMCI/REVIEW_Quantiles_LWEGAPS_ERA5_EVI_POP_discharge_InterpolatedNoise_monmean_detrend_2002-2019_allhotspots.csv", index=False)

scenario_list = [df_scenario2, df_scenario3, df_scenario4, df_scenario5, df_scenario6, df_scenario7]
table_list = []
table_scenario_observations_list = []
table_scenario_model_list = []
df_scenario_predictions_list = []
table_coefsign_list = []
table_lwe_bootstrap_list = []

for hotspot in hotspotnames:
    ### Calculate coefficients for baseline scenario ###
    df = df_scenario1[df_scenario1["hotspot"] == hotspot]
    data = df.values

    # Obtain previous timestep for each variable
    def generate_x(data):
        Z = np.zeros((len(data), 7))
        for t in range(0, len(data)):
            Z[t,1] = data[t, 3] # tp
            Z[t,3] = data[t, 6] # Population
        for t in range(1, len(data)):
            Z[t,0] = data[t-1, 2] # t2m-1
            Z[t,2] = data[t-1, 3] # tp-1
            Z[t,4] = data[t-1, 4] # evi-1
            Z[t,5] = data[t-1, 5] # lwe-1
            Z[t,6] = data[t-1, 7] # discharge-1
        Z = Z[1:,] # Remove first timestep
        return(Z)
    
    # X are variables from timestep t = 0 and -1, Y is lwe at t = 0
    X = generate_x(data) #[:,1:]
    y = data[1:,5] # lwe

    indices = np.arange(X.shape[0])

    ### Bootstrap NLS ###
    # Initialize list to store coefficients
    coefficients_list = []
    lwe_bootstrap_list = []

    # Perform bootstrapping 100 times
    for i in range(100):
        # Split training (80%) and test (20%) randomized
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        # Convert y_train to float type
        y_train = y_train.astype(float)
        X_train = X_train.astype(float)
        # Drop rows with NA values in y_train and corresponding rows in X_train
        na_mask = np.isnan(y_train) | np.isnan(X_train).any(axis=1)
        X_train = X_train[~na_mask]
        y_train = y_train[~na_mask]
        # Resample the training data with replacement
        X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
        # Perform regression on the resampled data
        coefficients, model = NLS(X_resampled, y_resampled)
        t2m_full, tp_full, tp_1_full, pop_full, evi_1_full, lwe_1_full, Q_1_full = X_to_vars(X)
        lwe_sample = func3(coefficients, t2m_full, tp_full, tp_1_full, pop_full, evi_1_full, lwe_1_full, Q_1_full)
        # Rerun with first timestep as initial value
        i = 1
        lwe_simulated_full_list = []
        for i in range(i, len(lwe_1_full)):
            if i == 1:
                lwe_simulated_full = np.array(lwe_1_full[i])
            else:
                lwe_simulated_full = func4(coefficients, t2m_full, tp_full, tp_1_full, pop_full, evi_1_full, lwe_simulated_full, Q_1_full, i)
            lwe_simulated_full_list.append(lwe_simulated_full.item())
        lwe_sample_2 = pd.Series(lwe_simulated_full_list)
        lwe_bootstrap_list.append(lwe_sample_2)
        coefficients_list.append(coefficients)

    # drop NA
    y_nona = y.astype(float)
    X_nona = X.astype(float)
    # Drop rows with NA values in y_train and corresponding rows in X_train
    na_mask = np.isnan(y_nona) | np.isnan(X_nona).any(axis=1)
    X_nona = X_nona[~na_mask]
    y_nona = y_nona[~na_mask]
    # Predict model on full data
    alpha_optimized, model_optimized = NLS(X_nona, y_nona)
    t2m_full, tp_full, tp_1_full, pop_full, evi_1_full, lwe_1_full, Q_1_full = X_to_vars(X)
    lwe_optimized = func3(alpha_optimized, t2m_full, tp_full, tp_1_full, pop_full, evi_1_full, lwe_1_full, Q_1_full)
    
    # Predict model on full data scenario 1 LWE simulated
    i = 1
    lwe_simulated_full_list = []
    for i in range(i, len(lwe_1_full)):
        if i == 1:
            lwe_simulated_full = np.array(lwe_1_full[i])
        else:
            lwe_simulated_full = func4(alpha_optimized, t2m_full, tp_full, tp_1_full, pop_full, evi_1_full, lwe_simulated_full, Q_1_full, i)
        lwe_simulated_full_list.append(lwe_simulated_full.item())
    lwe_predicted_2 = pd.Series(lwe_simulated_full_list)

    # Calculate mean and variance of coefficients    
    coefficients_array = np.array(coefficients_list)
    variance_coefficients = np.var(coefficients_array, axis=0)

    # Make dataframe of lwe_bootstrap per hotspot
    df_lwe_bootstrap = pd.DataFrame(lwe_bootstrap_list).T
    df_lwe_bootstrap["Observed"] = y[1:]
    df_lwe_bootstrap["Model"] = lwe_predicted_2
    df_lwe_bootstrap["Hotspot"] = hotspot
    df_lwe_bootstrap["time"] = df["time"].values[2:]
    table_lwe_bootstrap_list.append(df_lwe_bootstrap)

    # Calculate the standard errors using the standard deviation of residuals divided by the square root of the sample size
    std_dev_coef = np.sqrt(variance_coefficients)
    sample_size = len(X_resampled)
    SE_coef = std_dev_coef / np.sqrt(sample_size)
    degrees_of_freedom = sample_size - len(alpha_optimized)
    
    # Compute the t-values and p-values using the standard deviation method standard errors
    t_values = alpha_optimized / SE_coef
    p_values = [2 * (1 - t.cdf(np.abs(t_val), degrees_of_freedom)) for t_val in t_values]

    # Table significance coefficients
    table_coefsign = []
    for i, (coef, se, t_val, p_val) in enumerate(zip(alpha_optimized, SE_coef, t_values, p_values)):
        if p_val < 0.01:
            significance = "***"
        elif 0.01 <= p_val < 0.05:
            significance = "**"
        else:
            significance = ""
        table_coefsign.append([f"alpha_{i}", coef, se, t_val, p_val, significance])
    headers = ["Coefficient", "Value", "Std Error", "t-value", "p-value", "Significance"]
    table_df = pd.DataFrame(table_coefsign, columns=headers)
    table_df["Hotspot"] = hotspot
    table_df = table_df.round({"Value": 4, "Std Error": 4, "t-value": 4, "p-value": 4})
    table_coefsign_list.append(table_df)

    scenario_predictions = []
    for df_scenario in scenario_list:
        df_intv = df_scenario[df_scenario["hotspot"] == hotspot]
        data_intv = df_intv.values

        # X are variables from timestep t = 0 and -1, Y is lwe at t = 0
        X = generate_x(data) #[:,1:]
        y = data[1:,5] # lwe
        Xintv = generate_x(data_intv) #[:,1:]
        yintv = data_intv[1:,5] # lwe

        indices = np.arange(X.shape[0])

        # Split training (80%) and test (20%) randomized
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        Xintv_train, Xintv_test = Xintv[train_indices], Xintv[test_indices]
        yintv_train, yintv_test = yintv[train_indices], yintv[test_indices]

        # Define variables
        t2m_1 = np.array(X_train[:,0], dtype=float)
        t2m_1_test = np.array(X_test[:,0], dtype=float)
        t2m_full = np.array(X[:,0], dtype=float)
        t2m_intv_full = np.array(Xintv[:,0], dtype=float)
        tp = np.array(X_train[:,1], dtype=float)
        tp_test = np.array(X_test[:,1], dtype=float)
        tp_full = np.array(X[:,1], dtype=float)
        tp_intv_full = np.array(Xintv[:,1], dtype=float)
        tp_1 = np.array(X_train[:,2], dtype=float)
        tp_1_test = np.array(X_test[:,2], dtype=float)
        tp_1_full = np.array(X[:,2], dtype=float)
        tp_1_intv_full = np.array(Xintv[:,2], dtype=float)
        pop = np.array(X_train[:,3], dtype=float)
        pop_test = np.array(X_test[:,3], dtype=float)
        pop_full = np.array(X[:,3], dtype=float)
        pop_intv_full = np.array(Xintv[:,3], dtype=float)
        evi_1 = np.array(X_train[:,4], dtype=float)
        evi_1_test = np.array(X_test[:,4], dtype=float)
        evi_1_full = np.array(X[:,4], dtype=float)
        evi_1_intv_full = np.array(Xintv[:,4], dtype=float)
        lwe_1 = np.array(X_train[:,5], dtype=float)
        lwe_1_test = np.array(X_test[:,5], dtype=float)
        lwe_1_full = np.array(X[:,5], dtype=float)
        lwe_1_intv_full = np.array(Xintv[:,5], dtype=float)
        lwe = np.array(y_train, dtype=float)
        lwe_test = np.array(y_test, dtype=float)
        lwe_full = np.array(y, dtype=float)
        lwe_intv_full = np.array(yintv, dtype=float)
        Q_1 = np.array(X_train[:,6], dtype=float)
        Q_1_test = np.array(X_test[:,6], dtype=float)
        Q_1_full = np.array(X[:,6], dtype=float)
        Q_1_intv_full = np.array(Xintv[:,6], dtype=float)

        # Predict model on full data scenario X
        i = 1
        lwe_simulated_full_list = []
        for i in range(i, len(lwe_1_intv_full)):
            if i == 1:
                lwe_simulated_full = np.array(lwe_1_intv_full[i])
            else:
                lwe_simulated_full = func4(alpha_optimized, t2m_intv_full, tp_intv_full, tp_1_intv_full, pop_intv_full, evi_1_intv_full, lwe_simulated_full, Q_1_intv_full, i)
            lwe_simulated_full_list.append(lwe_simulated_full.item())
        lwe_predicted_3 = pd.Series(lwe_simulated_full_list)
        scenario_predictions.append(lwe_predicted_3)

    # Save scenario_predictions
    df_scenario_predictions = pd.DataFrame(scenario_predictions).T
    df_scenario_predictions.columns = [scenario2, scenario3, scenario4, scenario5, scenario6, scenario7]
    df_scenario_predictions["Model"] = lwe_predicted_2
    df_scenario_predictions["Hotspot"] = hotspot
    df_scenario_predictions_list.append(df_scenario_predictions)

    # Evaluate the model: Observed vs Model scenario 1 LWE simulated
    na_mask = np.isnan(lwe_full[test_indices].astype(float)) == False
    mse = mean_squared_error(lwe_full[test_indices][na_mask], lwe_predicted_2[test_indices][na_mask])
    r2 = r2_score(lwe_full[test_indices][na_mask], lwe_predicted_2[test_indices][na_mask])
    mae = mean_absolute_error(lwe_full[test_indices][na_mask], lwe_predicted_2[test_indices][na_mask])
    
    # Residuals analysis
    na_mask = np.isnan(lwe_full[1:,].astype(float)) == False
    residuals  = lwe_full[1:,][na_mask] - lwe_predicted_2[na_mask]
    mean_residuals = np.mean(residuals)
    stdev_residuals = np.std(residuals)
    stat, p_value = shapiro(residuals)
    residualsPandas = pd.DataFrame(residuals)
    values = pd.DataFrame(residualsPandas.values)
    dataframe = pd.concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t', 't+1'] 
    lagcorr_residuals = dataframe.corr().values[0,1]
    
    # Summarize model evaluation and residuals
    metrics = pd.DataFrame({'r²': [r2],'MAE': [mae],'MSE': [mse], "Mean residuals" : [mean_residuals], "Stdev residuals" : [stdev_residuals], "Lag correlation residuals":[lagcorr_residuals], "Shapiro stat": [stat], "Shapiro pvalue" : [p_value]}, index=[hotspot])
    metrics = metrics.round({'r²': 4, 'MAE': 4, 'MSE': 4, "Mean residuals" : 4, "Stdev residuals" : 4, "Lag correlation residuals": 4, "Shapiro stat": 4, "Shapiro pvalue" : 4})  
    alpha_list = ["alpha_" + str(i) for i in range(0, len(alpha_optimized))]
    coefs_dict = {var_name: alpha_optimized for var_name, alpha_optimized in zip(alpha_list, alpha_optimized)}
    coefs = pd.DataFrame(coefs_dict, index=[hotspot])
    table = pd.concat([metrics, coefs], axis=1)
    table_list.append(table)

    for i in range(0, len(scenario_predictions)):
        # Evaluate the model: Observed vs Model scenario X
        na_mask = np.isnan(lwe_full[test_indices].astype(float)) == False
        mse = mean_squared_error(lwe_full[test_indices][na_mask], scenario_predictions[i][test_indices][na_mask])
        r2 = r2_score(lwe_full[test_indices][na_mask], scenario_predictions[i][test_indices][na_mask])
        mae = mean_absolute_error(lwe_full[test_indices][na_mask], scenario_predictions[i][test_indices][na_mask])
        mbd = np.mean(scenario_predictions[i][test_indices][na_mask]- lwe_full[test_indices][na_mask])
        metrics = pd.DataFrame({'r²': [r2],'MAE': [mae],'MSE': [mse], "MBD":[mbd], "Scenario":[i+2]}, index=[hotspot])
        metrics = metrics.round({'r²': 4, 'MAE': 4, 'MSE': 4})
        table_scenario_observations_list.append(metrics)

        # Evaluate the model: Model scenario 1 LWE simulated vs Model scenario X
        mse = mean_squared_error(lwe_predicted_2[test_indices], scenario_predictions[i][test_indices])
        r2 = r2_score(lwe_predicted_2[test_indices], scenario_predictions[i][test_indices])
        mae = mean_absolute_error(lwe_predicted_2[test_indices], scenario_predictions[i][test_indices])
        mbd = np.mean(scenario_predictions[i][test_indices]- lwe_predicted_2[test_indices])
        metrics = pd.DataFrame({'r²': [r2],'MAE': [mae],'MSE': [mse], "MBD":[mbd], "Scenario":[i+2]}, index=[hotspot])
        metrics = metrics.round({'r²': 4, 'MAE': 4, 'MSE': 4})
        table_scenario_model_list.append(metrics)

table = pd.concat(table_list)
table 
table_scenario_observations = pd.concat(table_scenario_observations_list)
table_scenario_model = pd.concat(table_scenario_model_list)
scenario_predictions = pd.concat(df_scenario_predictions_list)
table_coefsign = pd.concat(table_coefsign_list)
#only keep unique rows from table_coefsign
table_coefsign = table_coefsign.drop_duplicates()
table_lwe_bootstrap = pd.concat(table_lwe_bootstrap_list, ignore_index=True)

# Display output tables
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
table # Performance metrics baseline models + residuals
table_scenario_observations # Performance metrics scenarios compared to observations
table_scenario_model # Performance metrics scenarios compared to baseline model
scenario_predictions # TWS values from modelled scenarios
table_coefsign # Significance of model coefficients
table_lwe_bootstrap # TWS values for 100 bootstrapped models

### Write CSV ###
table.to_csv("data/Output_StructuralCausalModelling/table_benchmark.csv", index=False, sep=';')
table_scenario_model.to_csv(f"data/Output_StructuralCausalModelling/table_scenario_model_{intervention}.csv", index=False)
table_scenario_observations.to_csv(f"data/Output_StructuralCausalModelling/table_scenario_observations_{intervention}.csv", index=False)
scenario_predictions.to_csv(f"data/Output_StructuralCausalModelling/{intervention}allhotspots.csv", index=False)
table_coefsign.to_csv("data/Output_StructuralCausalModelling/table_coefsignificance.csv", index=False, sep=";")
table_lwe_bootstrap.to_csv("data/Output_StructuralCausalModelling/table_lwe_predictions_bootstrap_model_observations.csv", index=False)