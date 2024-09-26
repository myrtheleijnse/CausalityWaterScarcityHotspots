########## Least Squares SCMs fitting and scenario modelling ##########
# Author: Myrthe Leijnse

### Imports ###
import os
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import shapiro
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

### Directory ###
os.getcwd()

### Reading Data ###
# Preprocessed
file_path = "data/Input_JPCMCI/Quantiles_LWE_ERA5_EVI_pop_ZonalInterpolated_monmean_2002-2019_allhotspots_scenario1.csv"
df_allhotspots = pd.read_csv(file_path)
hotspotnames = df_allhotspots['hotspot'].unique().tolist()

# Original
lwe_df = pd.read_csv("data/Input/LWE_2002-2023_ZonalInterpolated_hotspots_monthly.csv")
pr_df = pd.read_csv("data/Input/ERA5pr_1950-2023_Zonalmean_summed_hotspots.csv")
t2m_df = pd.read_csv("data/Input/ERA5t2m_1950-2022_Zonalmean_hotspots.csv")
pop_df = pd.read_csv("data/Input/pop_2000-2019_ZonalInterpolated_hotspots_monthly.csv")
EVI_df = pd.read_csv("data/Input/MODIS_EVI_irrarea_2001_2022_Zonalmmean_hotspots.csv")
Q_df = pd.read_csv("data/Input/discharge_1980-2019_Zonalmean_hotspots.csv")
dataframes = [lwe_df, pr_df, t2m_df, pop_df, EVI_df, Q_df]

df = dataframes[0]
for frame in dataframes[1:]:
    df = pd.merge(df, frame, on = ["time", "hotspot"], how = "inner")
df_intv = df.dropna()

### Functions ###
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

# NLS model functions
def func(alpha, t2m_1, tp):
    t2m_predicted = alpha[0] * t2m_1 + alpha[1] * tp + alpha[2]
    return t2m_predicted

def func6(alpha, evi_1, t2m_1, tp, pop, tp_1):
    evi_predicted = alpha[9] * evi_1 + alpha[10] * func(alpha, t2m_1, tp) + alpha[11] * pop + alpha[12] * tp + alpha[20] * tp_1 + alpha[13]
    return evi_predicted

def func7(alpha, Q_1, tp, tp_1):
    Q_predicted = alpha[15] * Q_1 + alpha[16] * tp + alpha[17] * tp_1 + alpha[18]
    return Q_predicted

def func2(alpha, t2m_1, tp, tp_1, pop, evi_1, lwe_1, lwe, Q_1):
    lwe_predicted = alpha[3] * func(alpha, t2m_1, tp) + alpha[4] * tp + alpha[5] * tp_1 + alpha[6] * pop + alpha[7] * func6(alpha, evi_1, t2m_1, tp, pop, tp_1) + alpha[8] + alpha[14] * lwe_1 + alpha[19] * func7(alpha, Q_1, tp, tp_1)
    residual = lwe_predicted - lwe
    return residual

def func3(alpha, t2m_1, tp, tp_1, pop, evi_1, lwe_1, Q_1):
    lwe_predicted = alpha[3] * func(alpha, t2m_1, tp) + alpha[4] * tp + alpha[5] * tp_1 + alpha[6] * pop + alpha[7] * func6(alpha, evi_1, t2m_1, tp, pop, tp_1) + alpha[8] + alpha[14] * lwe_1 + alpha[19] * func7(alpha, Q_1, tp, tp_1)
    return lwe_predicted

def func4(alpha, t2m_1, tp, tp_1, pop, evi_1, lwe_predicted, Q_1, i):
    lwe_predicted = alpha[3] * func(alpha, t2m_1[i-1:i], tp[i-1:i]) + alpha[4] * tp[i-1:i] + alpha[5] * tp_1[i-1:i] + alpha[6] * pop[i-1:i] + alpha[7] * func6(alpha, evi_1[i-1:i], t2m_1[i-1:i], tp[i-1:i], pop[i-1:i], tp_1[i-1:i]) + alpha[8] + alpha[14] * lwe_predicted + alpha[19] * func7(alpha, Q_1[i-1:i], tp[i-1:i], tp_1[i-1:i])
    return np.array(lwe_predicted)

### Execution ###
# Impact analysis: perturbations on input data: example of scenario -10% and -1K
degrees = -1
percentage = 0.9
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
    df_variable = normalize_data(df_variable_monavg, df_variable_monavg, variable, variable)
    df_variable_intv = normalize_data(df_variable_intv_monavg, df_variable_monavg, variable, variable_intv)
    df_qnorm = pd.concat([df_qnorm, df_variable[variable], df_variable_intv[variable_intv]], axis=1)

pop = df_intv[["time","hotspot","Population", "Population_intv"]]
df_pop = normalize_data(pop, pop, "Population", "Population")
df_pop_intv = normalize_data(pop, pop, "Population", "Population_intv")
df_qnorm = pd.concat([df_intv["hotspot"], df_intv["time"], df_qnorm, df_pop["Population"], df_pop_intv["Population_intv"]], axis=1)

# Scenario definition
scenario1 = "No intervention"
scenario2 = "-10% population"
scenario3 = "-1K temperature"
scenario4 = "-10% precipitation"
scenario5 = "-10% EVI"
scenario6 = "-10% LWE thickness"
scenario7 = "-10% discharge"

df_scenario1 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI", "lwe_thickness", "Population", "discharge"]]
df_scenario2 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI", "lwe_thickness", "Population_intv", "discharge"]]
df_scenario3 = df_qnorm[["hotspot", "time", "t2m_intv", "tp", "EVI", "lwe_thickness", "Population", "discharge"]]
df_scenario4 = df_qnorm[["hotspot", "time", "t2m", "tp_intv", "EVI", "lwe_thickness", "Population", "discharge"]]
df_scenario5 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI_intv", "lwe_thickness", "Population", "discharge"]]
df_scenario6 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI", "lwe_thickness_intv", "Population", "discharge"]]
df_scenario7 = df_qnorm[["hotspot", "time", "t2m", "tp", "EVI", "lwe_thickness", "Population", "discharge_intv"]]

# write csv of scenario1
df_scenario1.to_csv("data/Input_JPCMCI/Quantiles_LWE_ERA5_EVI_POP_discharge_Interpolated_monmean_2002-2019_allhotspots_scenario1.csv", index=False)

scenario_list = [df_scenario2, df_scenario3, df_scenario4, df_scenario5, df_scenario6, df_scenario7]
table_list = []
table_scenario_observations_list = []
table_scenario_model_list = []
df_scenario_predictions_list = []
for hotspot in hotspotnames:
    df = df_scenario1[df_scenario1["hotspot"] == hotspot]
    data = df.values
    scenario_predictions = []
    for df_scenario in scenario_list:
        df_intv = df_scenario[df_scenario["hotspot"] == hotspot]
        data_intv = df_intv.values

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
                Z[t,5] = data[t-1, 5] # TWS-1
                Z[t,6] = data[t-1, 7] # discharge-1
            Z = Z[1:,] # Remove first timestep
            return(Z)

        # X are variables from timestep t = 0 and -1, Y is TWS at t = 0
        X = generate_x(data) 
        y = data[1:,5] # TWS
        Xintv = generate_x(data_intv) 
        yintv = data_intv[1:,5] # TWS

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

        # Initial values alpha and beta coefficients
        initial = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # Fit model on training data scenario 1
        fit = least_squares(func2, initial, method='lm', args=(t2m_1, tp, tp_1, pop, evi_1, lwe_1, lwe, Q_1))
        alpha_optimized = fit.x
        
        # Predict model on full data scenario 1 TWS observed
        lwe_predicted_full = func3(alpha_optimized, t2m_full, tp_full, tp_1_full, pop_full, evi_1_full, lwe_1_full, Q_1_full)

        # Predict  model on full data scenario 1 TWS simulated
        i = 1
        lwe_simulated_full_list = []
        for i in range(i, len(lwe_1_full)):
            if i == 1:
                lwe_simulated_full = np.array(lwe_1_full[i])
            else:
                lwe_simulated_full = func4(alpha_optimized, t2m_full, tp_full, tp_1_full, pop_full, evi_1_full, lwe_simulated_full, Q_1_full, i)
            lwe_simulated_full_list.append(lwe_simulated_full.item())
        lwe_predicted_2 = pd.Series(lwe_simulated_full_list)

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

    # Plotting
    plt.figure()
    plt.plot(range(len(lwe_full)), lwe_full, label="Observed", color="pink")
    #plt.plot(range(len(lwe_predicted_full)), lwe_predicted_full, label="Model LWE observed", color="violet")
    plt.plot(range(len(lwe_predicted_2)), lwe_predicted_2, label="Model LWE simulated", color="#377eb8")
    plt.plot(range(len(scenario_predictions[0])), scenario_predictions[0], label=scenario2, color="#ff7f00")
    plt.plot(range(len(scenario_predictions[1])), scenario_predictions[1], label=scenario3, color="#4daf4a")
    plt.plot(range(len(scenario_predictions[2])), scenario_predictions[2], label=scenario4, color="#f781bf")
    plt.plot(range(len(scenario_predictions[3])), scenario_predictions[3], label=scenario5, color="#a65628")
    plt.plot(range(len(scenario_predictions[4])), scenario_predictions[4], label=scenario6, color="#984ea3")
    plt.plot(range(len(scenario_predictions[5])), scenario_predictions[5], label=scenario7, color="#999999")
    plt.legend()
    plt.title("Observed vs Predicted TWS")
    plt.suptitle(hotspot)
    plt.xlabel("Time")
    plt.ylabel("TWS")
    plt.show()

    # Save scenario_predictions
    df_scenario_predictions = pd.DataFrame(scenario_predictions).T
    df_scenario_predictions.columns = [scenario2, scenario3, scenario4, scenario5, scenario6, scenario7]
    # df_scenario_predictions["Observed"] = lwe_full
    df_scenario_predictions["Model"] = lwe_predicted_2
    df_scenario_predictions["Hotspot"] = hotspot
    df_scenario_predictions_list.append(df_scenario_predictions)

    # Evaluate the model: Observed vs Model scenario 1 TWS simulated
    mse = mean_squared_error(lwe_full[test_indices], lwe_predicted_2[test_indices])
    r2 = r2_score(lwe_full[test_indices], lwe_predicted_2[test_indices])
    mae = mean_absolute_error(lwe_full[test_indices], lwe_predicted_2[test_indices])
    
    # Residuals analysis
    residuals  = lwe_full[1:,] - lwe_predicted_2
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
        mse = mean_squared_error(lwe_full[test_indices], scenario_predictions[i][test_indices])
        r2 = r2_score(lwe_full[test_indices], scenario_predictions[i][test_indices])
        mae = mean_absolute_error(lwe_full[test_indices], scenario_predictions[i][test_indices])
        mbd = np.mean(scenario_predictions[i][test_indices]- lwe_full[test_indices])
        metrics = pd.DataFrame({'r²': [r2],'MAE': [mae],'MSE': [mse], "MBD":[mbd], "Scenario":[i+2]}, index=[hotspot])
        metrics = metrics.round({'r²': 4, 'MAE': 4, 'MSE': 4})
        table_scenario_observations_list.append(metrics)

        # Evaluate the model: Model scenario 1 TWS simulated vs Model scenario X
        mse = mean_squared_error(lwe_predicted_2[test_indices], scenario_predictions[i][test_indices])
        r2 = r2_score(lwe_predicted_2[test_indices], scenario_predictions[i][test_indices])
        mae = mean_absolute_error(lwe_predicted_2[test_indices], scenario_predictions[i][test_indices])
        mbd = np.mean(scenario_predictions[i][test_indices]- lwe_predicted_2[test_indices])
        metrics = pd.DataFrame({'r²': [r2],'MAE': [mae],'MSE': [mse], "MBD":[mbd], "Scenario":[i+2]}, index=[hotspot])
        metrics = metrics.round({'r²': 4, 'MAE': 4, 'MSE': 4})
        table_scenario_model_list.append(metrics)

table = pd.concat(table_list)
table_scenario_observations = pd.concat(table_scenario_observations_list)
table_scenario_model = pd.concat(table_scenario_model_list)
scenario_predictions = pd.concat(df_scenario_predictions_list)

# Display output tables
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
table # Performance metrics baseline models + residuals
table_scenario_observations # Performance metrics scenarios compared to observations
table_scenario_model # Performance metrics scenarios compared to baseline model
scenario_predictions # TWS values from modelled scenarios

### Write CSV ###
table.to_csv("/eejit/home/5738091/data/Causality/metrics_csv/table_baseline_model_performance.csv")
table_scenario_model.to_csv("data/Output/table_scenario_model_performance_-10perc.csv")
table_scenario_observations.to_csv("data/Output/table_scenario_observations_performance_-10perc.csv")
scenario_predictions.to_csv("data/Output/-10allhotspots.csv")