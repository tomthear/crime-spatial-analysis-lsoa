***

# **README**

## **Project: LSOA‑Level Crime Modelling, Spatial Analysis and Predictive Pipeline**

This project builds a complete, reproducible end‑to‑end workflow for analysing crime patterns at the LSOA level using street‑level crime data, Census 2021 indicators, population estimates, and geospatial boundary data. It includes ETL, feature engineering, diagnostics, predictive modelling, spatial econometrics, and panel modelling, producing interpretable outputs suitable for research, strategy, and policy analysis.

***

## **1. Objectives**

*   Construct a clean, unified analytical dataset linking:
    *   Crime records
    *   Census indicators
    *   Housing composition
    *   Population time series
    *   Spatial adjacency
*   Build transparent, interpretable statistical and ML models to identify structural correlates of crime.
*   Evaluate predictive generalisation using LAD‑grouped CV and hold‑out tests.
*   Implement spatial econometric models (SAR, SEM) to correct for autocorrelation.
*   Run multi‑year panel Negative Binomial models for crime counts.
*   Export reproducible outputs for further mapping (GIS, Power BI) and reporting.

***

## **2. Data Sources**

*   **Street-level crime** (Home Office).
*   **Outcomes data** (Home Office).
*   **Census 2021: TS067, TS044** (ONS).
*   **Population estimates 2015–2022** (https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates).
*   **ONS LSOA/MSOA/LAD lookup** (PCD OA21 → LSOA/MSOA/LAD).
*   **ONS LSOA 2021 boundaries** (https://borders.ukdataservice.ac.uk/easy_download_data.html?data=England_lsoa_2021).
*   **Administrative Areas** (https://www.doogal.co.uk/AdministrativeAreas).

***

## **3. Pipeline Overview**

### **ETL & Data Preparation**

0.  Pre-processing steps - detailed within the notebook first cell
1.  Import libraries and configure environment.
2.  Load raw crime, census, population, and geospatial files.
3.  Build modelling tables: `mod_all`, `X_all`, `y_all`, `groups`.
4.  Clean missing/outlier values, apply winsorisation.
5.  Diagnose and prune features (rank tests, VIF, collinearity screens).

### **Train/Test Discipline**

6.  Perform LAD‑grouped train/test split (prevents geographic leakage).
7.  Run GroupKFold CV on training set only.
8.  Fit final predictive model on training data only.
9.  Evaluate once on the hold‑out test set.

### **Full-Data Modelling**

10. Fit OLS with HC3 robust errors.
11. Run full diagnostics (residuals, influence, Cook’s D, QQ, scale–location).
12. Fit spatial models (SAR, SEM) with queen‑contiguity/KNN weights.
13. Fit other models: ElasticNet, panel Negative Binomial.

### **Exports**

14. Write out:
    *   cleaned datasets
    *   model outputs
    *   spatial metrics
    *   aligned GeoJSONs
    *   tables for Power BI / GIS
    *   diagnostic images

***

## **4. Modelling Methods**

### **Predictive Models**

*   OLS (with and without HC3 robust errors)
*   Penalised regression (ElasticNetCV)
*   LAD‑grouped CV and hold‑out evaluation
*   StandardScaler for continuous features
*   Coefficient importance, partial regression, and stability checks

### **Spatial Econometric Models**

*   **SAR (Spatial Lag Model)**
*   **SEM (Spatial Error Model)**
*   LSOA adjacency built with queen contiguity + KNN fallback
*   Moran’s I residual diagnostics
*   Direct, indirect, and total impact calculations

### **Panel Models**

*   Negative Binomial GLM with population offset for yearly crime counts
*   Stacked panel (2015–2022)
*   Year‑invariant and structural predictors

***

## **5. Key Outputs**

*   Unified dataset with:
    *   population (2015–2022)
    *   crime counts & crime rates (2015–2022)
    *   accommodation types (2021, per‑10k)
    *   qualifications (2021, per‑10k)
    *   under‑16 population
*   Spatial weights matrix
*   OLS, HC3, SAR, SEM model summaries
*   Cross-validation metrics
*   Test‑set predictive performance
*   Residual and influence diagnostics
*   GeoJSON boundaries with merged model outputs
*   Panel NB regression summary
*   Visualisations (correlation heatmaps, PCA plots, Cook’s D, choropleths)
*   Output dataset

***

## **6. Reproducibility & Structure**

All outputs are saved under timestamped directories to ensure deterministic runs.

Typical folder layout:

    project/
    │
    ├── notebooks/
    ├── data_raw/
    ├── data_intermediate/
    ├── data_outputs_YYYYMMDD_HHMMSS/
    │   ├── cleaned_data.csv
    │   ├── model_table_aligned.csv
    │   ├── lsoa_subset.geojson
    │   ├── spatial_metrics.csv
    │   ├── diagnostics/
    │   └── images/
    └── README.md

***

## **7. Requirements**

*   Python ≥ 3.9
*   Libraries:
    *   pandas, numpy, geopandas
    *   statsmodels, sklearn
    *   libpysal, esda, splot
    *   matplotlib, seaborn
    *   scipy

***

## **8. Intended Use & Ethical Guidance**

This project produces **area‑level**, **interpretable** models.  
It is suitable for:

*   research
*   policy analysis
*   environmental/situational crime prevention
*   resource allocation at neighbourhood level

It **must not** be used for person‑level prediction, profiling, or individual targeting.

***

## **9. License & Attribution**

*   Data © Crown copyright / ONS / Home Office (as per respective terms).
*   Code and analytical pipeline authors own

***
