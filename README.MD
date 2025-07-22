# Translation Efficiency Covariation (TEC) Network Analyses

This repository contains code supporting the manuscript titled **Topological and Functional Characterization of Human Translation Efficiency Covariation Network**. It includes scripts used for:
- Characterization of the TEC network
- Comparative analysis with the corresponding RNA co-expression network
- Functional analysis of the TEC network
- Analysis of tissue-specific TEC networks
- Supplementary and supporting analyses

## Overview
The following Jupyter notebooks correspond to the four results subsections of the manuscript:
1. `tec_characterization.ipynb` - Characterization of TEC network 
2. `rna_comparison.ipynb` - Comparison between the TEC network and the RNA co-expression network; includes Supplementary Figure 2a and Table 3
3. `tec_biology.ipynb` - Functional analysis and clustering of the TEC network
4. `tissue_net.ipynb` - Analysis of the liver-specific TEC network
5. `supplemental/` - All supplemental analyses, figures, and tables generation

## Workflow
To run any of the analyses, open the corresponding `.ipynb` file and select **Run All**.

If you wish to save the output figures, please create the following directories beforehand: 
- `figure/` - main figures
- `supplemental/supp_fig/` - supplementary figures

## Dependencies
This project uses **Python 3.12**.

To install the required packages, run:
```
pip install -r requirements.txt
```

## Data
<!-- The `./data` directory contains most of the data used throughout this study. More information can be found within the README file within the directory.  -->
Due to size limitations, data files are not hosted on GitHub. They can be found via this [Box Link](https://utexas.box.com/s/ifelzahwvsmmhwhs77skt5mmfn3eutaf)

## Contact
If you have questions, please email hurleyqi@utexas.edu.