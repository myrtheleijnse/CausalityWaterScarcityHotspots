![Dependabot](https://img.shields.io/github/languages/code-size/myrtheleijnse/2_causality)

# Welcome :star2:
Project: Causality and water scarcity hotspots

Author: M. Leijnse

Contact: m.leijnse@uu.nl

Organization: Utrecht University

## Introduction
Hello, thank you for reading me.

This repository consist of scripts and data used for the National Geographic World Water Map project. 

## Download
- Install Python version 3.9.12
- Install Packages numpy, os, pandas, matplotlib, scipy, sklearn, tigramite

Download
```
git clone git@github.com:myrtheleijnse/CausalityWaterScarcityHotspots.git
```

## Project organization
- PG = project-generated
- HW = human-writable
- RO = read only
```
.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── scripts						<- All scripts used to generate output
├── data               			<- All project data
│   ├── Input      				<- Raw input data
│   ├── Input_JPCMCI           	<- Preprocessed input data
│   └── Output_ImpactAnalysis   <- Output data of model performance metrics and scenario results (performance)
└── docs               			<- Documentation
   └── manuscript     			<- Manuscript source (RO)
   └── supplementary        <- Other project outputs (RO)

```


## License
This project is licensed under the terms of the [MIT License](/LICENSE.md)
