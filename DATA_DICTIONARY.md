# **DATA\_DICTIONARY.md**

## **Overview**

This data dictionary describes all fields in the unified LSOA‑level dataset created during the crime‑modelling pipeline.  
It covers identifiers, population and census variables, crime counts/rates, engineered features, and spatial modelling fields.

***

## **1. Geographic Identifiers**

| Field                           | Type   | Description                                                             |
| ------------------------------- | ------ | ----------------------------------------------------------------------- |
| `geography`                     | string | LSOA name (e.g., *Portsmouth 012B*).                                    |
| `geography_code` / `lsoa21cd`   | string | 2021 LSOA code, primary key (e.g., *E01012345*).                        |
| `msoa21cd`                      | string | 2021 MSOA code for the LSOA.                                            |
| `ladcd`                         | string | Local Authority District code; also used as group label for grouped CV. |
| `lsoa21nm`, `msoa21nm`, `ladnm` | string | Human‑readable names for LSOA/MSOA/LAD.                                 |

***

## **2. Population (2015–2022)**

*All population fields are integer counts.*

| Field             | Type | Description                                                 |
| ----------------- | ---- | ----------------------------------------------------------- |
| `Population_YYYY` | int  | Mid‑year ONS population estimate for year YYYY (2015–2022). |
| `Under_16_2021`   | int  | Derived: population under 16 for Census 2021.               |

***

## **3. Crime Counts (2015–2022)**

| Field              | Type | Description                                                   |
| ------------------ | ---- | ------------------------------------------------------------- |
| `crime_count_YYYY` | int  | Recorded crime count for year YYYY, aggregated at LSOA level. |

***

## **4. Crime Rates (per 10,000 population)**

| Field                     | Type  | Description                                                                   |
| ------------------------- | ----- | ----------------------------------------------------------------------------- |
| `crime_rate_YYYY_per_10k` | float | Crime rate per 10,000 residents for year YYYY. Used as targets for modelling. |

Formula:

    crime_rate = (crime_count / population) * 10,000

***

## **5. Census 2021 — Accommodation (per 10k population)**

| Field                                 | Type  | Description                                        |
| ------------------------------------- | ----- | -------------------------------------------------- |
| `Detached_per_10k_2021`               | float | Population living in detached housing.             |
| `Semi_detached_per_10k_2021`          | float | Population in semi‑detached houses.                |
| `Terraced_per_10k_2021`               | float | Population in terraced housing.                    |
| `Purpose_built_flats_per_10k_2021`    | float | Population in purpose‑built flats/tenements.       |
| `Converted_shared_house_per_10k_2021` | float | Converted/shared houses.                           |
| `Converted_other_per_10k_2021`        | float | Other converted buildings.                         |
| `Commercial_building_per_10k_2021`    | float | Residents in commercial premises.                  |
| `Caravan_mobile_per_10k_2021`         | float | Residents in caravans/mobile/temporary structures. |

***

## **6. Census 2021 — Qualification Levels (per 10k population)**

| Field                               | Type  | Description                                 |
| ----------------------------------- | ----- | ------------------------------------------- |
| `No_qualifications_per_10k_2021`    | float | Residents with no qualifications.           |
| `Level_1_per_10k_2021`              | float | Level 1 qualifications.                     |
| `Level_2_per_10k_2021`              | float | Level 2 qualifications.                     |
| `Level_3_per_10k_2021`              | float | Level 3 qualifications.                     |
| `Level_4_plus_per_10k_2021`         | float | Degree or higher.                           |
| `Apprenticeship_per_10k_2021`       | float | Apprenticeship‑level qualifications.        |
| `Other_qualifications_per_10k_2021` | float | Other or foreign‑equivalent qualifications. |

***

## **7. Engineered Modelling Fields**

| Field                  | Type      | Description                                                     |
| ---------------------- | --------- | --------------------------------------------------------------- |
| `y_all`                | Series    | Target vector (e.g., crime rate for chosen year).               |
| `X_all`                | DataFrame | Feature matrix used for modelling.                              |
| `groups`               | string    | LAD‑based grouping variable for non‑leakage CV.                 |
| `mod_all`              | DataFrame | Fully cleaned modelling dataset (census × populations × crime). |
| `winsorised_columns`   | list      | Columns capped using z‑score‑based winsorisation.               |
| `z_score_outlier_flag` | bool      | Indicates >3 SD outliers (for diagnostics only).                |

***

## **8. Spatial Modelling Fields**

| Field                                 | Type          | Description                                        |
| ------------------------------------- | ------------- | -------------------------------------------------- |
| `W`                                   | sparse matrix | Spatial weights matrix (queen contiguity + KNN).   |
| `moran_I`                             | float         | Moran’s I statistic for spatial autocorrelation.   |
| `moran_p_norm`                        | float         | p‑value for Moran’s I (normal reference).          |
| `fitted_ols`                          | float         | OLS fitted values for mapping/spatial diagnostics. |
| `resid_ols`                           | float         | OLS residuals.                                     |
| `resid_sar`                           | float         | SAR residuals.                                     |
| `sar_aic`, `sar_bic`, `sar_pseudo_r2` | float         | Spatial lag model performance metrics.             |
| `ols_aic`, `ols_bic`, `ols_r2`        | float         | Baseline OLS fit metrics.                          |

***

## **9. Panel Model (Long Format) Fields**

Used only in Negative Binomial panel model.

| Field   | Type | Description                |
| ------- | ---- | -------------------------- |
| `year`  | int  | Year (2015–2022).          |
| `count` | int  | Crime count in that year.  |
| `pop`   | int  | Population offset for GLM. |

***

## **10. Exported Files (Reference Only)**

| File                               | Description                             |
| ---------------------------------- | --------------------------------------- |
| `crime_final.csv`                  | Cleaned crime + outcomes dataset.       |
| `crime_aggregated.csv`             | LSOA × year aggregation.                |
| `census_with_population.csv`       | Census merged with population.          |
| `census_with_crime_rates.csv`      | Final modelling dataset.                |
| `model_table_aligned.csv`          | Spatially aligned table for PySAL.      |
| `lsoa_subset.geojson`              | Boundary file trimmed to used LSOAs.    |
| `spatial_metrics.csv`              | Spatial model evaluation.               |
| `elasticnet_selected_features.csv` | Features retained after regularisation. |

