########## Joint-PCMCI multiple hotspots and Expert Graph ##########
# Author: Myrthe Leijnse

### Imports ###
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

import tigramite
from tigramite.jpcmciplus import JPCMCIplus
from tigramite.independence_tests.parcorr_mult import ParCorrMult
import tigramite.data_processing as pp
import tigramite.plotting as tp

### Directory ###
os.getcwd()

### Load data ###

df_allhotspots = pd.read_csv(r"data/Input_JPCMCI/REVIEW_Quantiles_LWEGAPS_ERA5_EVI_POP_discharge_InterpolatedNoise_monmean_detrend_2002-2019_allhotspots.csv")

hotspotnames = df_allhotspots['hotspot'].unique().tolist()

## Select variables ##
df_allhotspots.columns.values
var_names = ["lwe_thickness", "tp", "t2m", "EVI", "Population", "discharge"]

## Select hotspot ##
list_hotspots = hotspotnames 
figname = "".join(list_hotspots)
data_observed = {}
N = 1
for hotspot in list_hotspots:
    df = df_allhotspots[df_allhotspots["hotspot"] == hotspot]
    df = df[var_names]
    data = df.values
    data_observed[N] = data
    N= N + 1
   
### Initialize dataframe ###
dataframe = pp.DataFrame(
    data=data_observed,
    analysis_mode = 'multiple',
    var_names = var_names,
    missing_flag = 999.,
    )

# plot dataframe
tp.plot_timeseries(dataframe)

### Node classification ###
node_classification = {
    0: "system",
    1: "system",
    2: "system",
    3: "system",
    4: "system",
    5: "system",
    6: "system"
}
observed_indices = [0,1,2,3,4,5,6]
node_classification_jpcmci = {i: node_classification[var] for i, var in enumerate(observed_indices)}

### J-PCMCI+ ###
jpcmciplus = JPCMCIplus(dataframe=dataframe,
                          cond_ind_test=ParCorrMult(significance='analytic'), 
                          verbosity=1,
                          node_classification=node_classification_jpcmci)
results = jpcmciplus.run_jpcmciplus(tau_min=0, 
                              tau_max=11, 
                              pc_alpha=0.001)

var_names = ["TWS", "pr", "t2m", "EVI", "pop", "Q"]

### Plotting ###
tp.plot_graph(
    results['graph'], 
    val_matrix=results['val_matrix'], 
    var_names=var_names, 
    link_colorbar_label="Link Strength", 
    node_colorbar_label="Node Strength"
)
tp.plot_graph(results['graph'], val_matrix=results['val_matrix'], var_names=var_names, link_colorbar_label = "Link Strength", node_colorbar_label = "Node Strength")

### Expert Graph ###
print(results["graph"])

expert_graph = np.array([[['', '-->', '', '-->', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', '']], #TWS

       [['-->', '-->', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', ''],
        ['-->', '', '', '', '', '', '', '', '', '', '', ''],
        ['-->', '-->', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', ''],
        ['-->', '', '', '', '', '', '', '', '', '', '', '']], #pr

       [['-->', '', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '-->', '-->', '', '', '', '', '', '', '', '', ''],
        ['-->', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', ''],
        ['-->', '', '', '', '', '', '', '', '', '', '', '']], #t2m

       [['-->', '', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '<--', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '-->', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', '']], #EVI

        [['-->', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', ''],
        ['-->', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '-->', '-->', '-->', '', '', '', '', '', '', '', ''],
        ['-->', '', '', '', '', '', '', '', '', '', '', '']], #pop

       [['-->', '', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', ''],
        ['<--', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '-->', '', '', '', '', '', '', '', '', '', '']]], #Q
      dtype='<U3')

var_names = ["TWS", "pr", "t2m", "EVI", "pop", "Q"]

tp.plot_graph(expert_graph, var_names=var_names)