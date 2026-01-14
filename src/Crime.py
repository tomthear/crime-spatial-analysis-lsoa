#!/usr/bin/env python
# coding: utf-8

# # Installs

# In[ ]:


# get_ipython().system('pip install libpysal')


# In[ ]:


# # Refine!
# pip install -r requirements.txt


# # Pre-process step.  
# Population data has been pre-processed before loading.  The data has been downloaded (https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates/mid2022revisednov2025tomid2024/sapelsoasyoa20222024.xlsx) and for each year a sum has been created that totals females aged 0-99 and males ages 0-90 (this is a limitation of the data).  These are added in columns labelled Male XXXX, Female XXXX (where XXXX is the year).  A new table was created, which compiled these with LAD 2021 code, LAD 2021 Name, and then for each year Total XXXX, Female XXXX and Male XXXX.  This is then saved as Population.csv within eg: C:\Users\ZZZZZ\Crime\Source\Population.  

# In[1]:


get_ipython().system('pip install libpysal')
get_ipython().system('pip install spreg')
get_ipython().system('pip install geopandas')
get_ipython().system('pip install esda')
get_ipython().system('pip install splot')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


get_ipython().system('pip install libpysal')
get_ipython().system('pip install spreg')
get_ipython().system('pip install geopandas')
get_ipython().system('pip install esda')
get_ipython().system('pip install splot')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


import warnings
warnings.filterwarnings('ignore')rom IPython.display import Markdown
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import sweetviz as sv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from libpysal import weights
from esda.moran import Moran
from splot.esda import moran_scatterplot
from spreg import ML_Lag, ML_Error
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline


# In[2]:


import warnings
warnings.filterwarnings('ignore')rom IPython.display import Markdown
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import sweetviz as sv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from libpysal import weights
from esda.moran import Moran
from splot.esda import moran_scatterplot
from spreg import ML_Lag, ML_Error
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline

# from sklearn.cluster import KMeans
# from scipy.sparse import identity as sp_identity
# from scipy.sparse.linalg import spsolve
# import shutil
# import libpysal
# from libpysal.weights import Queen, KNN


# ## Plot style defaults
# Matplotlib and Seaborn palette.  Set default colour maps.

# In[3]:


# # Matplotlib defaults 
# plt.rcParams["image.cmap"] = "cividis"  # colour-blind safe, perceptually uniform
# plt.rcParams["axes.prop_cycle"] = plt.cycler(
#     color=plt.cm.cividis(np.linspace(0, 1, 10))
# )

# # Improve readability
# plt.rcParams["axes.facecolor"] = "#FFFFFF"
# plt.rcParams["figure.facecolor"] = "#FFFFFF"
# plt.rcParams["axes.edgecolor"] = "#333333"
# plt.rcParams["grid.color"] = "#D0D0D0"

# # Seaborn defaults 
# sns.set_theme(
#     style="whitegrid",
#     palette="colorblind"   # Tol colour-blind-safe palette
# )


# In[4]:


# Universal figure style
plt.rcParams.update({
    "figure.figsize": (7, 5),
    "figure.dpi": 130,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.6,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.color": "#D3D3D3",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
})

# Accessible continuous colormap (for heatmaps, maps, diagnostics)
plt.rcParams["image.cmap"] = "viridis"  # or "cividis" or "plasma"

# Accessible categorical palette
sns.set_palette("colorblind")  # Tol palette

# Seaborn global style
sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.0,
    rc={
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": True,
    }
)

# For GeoPandas maps (soft borders)
map_edge = {"linewidth": 0.2, "edgecolor": "#e6e6e6"}  # use with gdf.plot(**map_edge)


# # PARAMETERS

# In[5]:


#Root folder of project
BASEROOT = Path(r"C:\Users\cxputte\Crime")
#Job start time
JOB_START_FORMAT = (datetime.now().strftime("%Y%m%d_%H%M%S_"))
#Define outputs file and create it
OUTPUTS_DIRNAME = f"{JOB_START_FORMAT}Outputs"
OUTPUTS_PATH = BASEROOT / OUTPUTS_DIRNAME
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
#Data folder
ROOT = Path(r"C:\Users\cxputte\Crime\Source")
CRIME_YEARS_DIRNAME = "Crime_Years"
#LSOA data
LSOA_MSOA_LAD_DIRNAME = "LSOA_MSOA_LAD"
LSOA_MSOA_LAD_FILENAME = "PCD_OA21_LSOA21_MSOA21_LAD_NOV24_UK_LU.csv"
#Population data
POPULATION_DIRNAME = "Population"
POPULATION_FILENAME = "Population.csv"
#Census data
CENSUS_DIRNAME = "Census"
CENSUS_FILENAMES = ["census2021-ts067-lsoa.csv", "census2021-ts044-lsoa.csv"]
#Crime years
CRIME_YEARS = ROOT / CRIME_YEARS_DIRNAME
STREET_DIRNAME = "street_files" 
OUTCOMES_DIRNAME = "outcomes_files"
STREET_DIR = ROOT / CRIME_YEARS_DIRNAME/ STREET_DIRNAME
OUTCOMES_DIR = ROOT / CRIME_YEARS_DIRNAME/ OUTCOMES_DIRNAME
#Filepaths
LSOA_MSOA_LAD_FILEPATH = ROOT/LSOA_MSOA_LAD_DIRNAME/LSOA_MSOA_LAD_FILENAME
POPULATION_FILEPATH = ROOT/POPULATION_DIRNAME/POPULATION_FILENAME
CENSUS_DIR = ROOT / CENSUS_DIRNAME


# In[6]:


# Old params:
# CENSUS_FILEPATH = ROOT/CENSUS_DIRNAME/CENSUS_FILENAMES
# TARGET_DIRS = [STREET_DIR, OUTCOMES_DIR] #Only used within the file moving section
# DRY_RUN = False  # set True to preview actions without copying
# STREET_SUFFIX = "-street.csv" #Only used within the file moving section
# OUTCOMES_SUFFIX = "-outcomes.csv"#Only used within the file moving section

# OUTPUTS_PATH= OUTPUTS_PATH / (datetime.now().strftime("%Y%m%d_%H%M%S_"))


# ## Create LSOA mapping table for ones we are interested in
# 
# As the dataset is huge (50,460,858 street level crimes), decision made to look at Hampshire and its closest stat-neighbour Wiltshire as a comparison.  Optional expansion to top 5 statistical neighbours if time allows.

# In[7]:


print(OUTPUTS_PATH / (datetime.now().strftime("%Y%m%d_%H%M%S_")))


# In[8]:


# Columns to keep
keep_cols = ["lsoa21cd", "msoa21cd", "ladcd", "lsoa21nm", "msoa21nm", "ladnm", "ladnmw"]
# Load CSV
df_LSOA_MSOA_LAD = pd.read_csv(LSOA_MSOA_LAD_FILEPATH, low_memory=False, encoding="cp1252")

# Keep only required columns
df_LSOA_MSOA_LAD = df_LSOA_MSOA_LAD[keep_cols]

#Drop duplicates
distinct_all = df_LSOA_MSOA_LAD.drop_duplicates()
print(distinct_all)

output_path_pairs = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_distinct_all.csv"
distinct_all.to_csv(output_path_pairs, index=False)
print(f"Saved LSOA code name pairs to: {output_path_pairs}")


# In[9]:


# LAD codes to filter
lad_codes = [
                #0
            "E07000084",                         # Basingstoke and Deane (Hampshire)
            "E07000085",                         # East Hampshire (Hampshire)
            "E07000086",                         # Eastleigh (Hampshire)
            "E07000087",                         # Fareham (Hampshire)
            "E07000088",                         # Gosport (Hampshire)
            "E07000089",                         # Hart (Hampshire)
            "E07000090",                         # Havant (Hampshire)
            "E07000091",                         # New Forest (Hampshire)
            "E07000092",                         # Rushmoor (Hampshire)
            "E07000093",                         # Test Valley (Hampshire)
            "E07000094",                         # Winchester (Hampshire)

            "E06000044",                         # Portsmouth Unitary (Hampshire)
            "E06000045",                         # Southampton Unitary (Hampshire)

                #1
            "E06000054",                          # Wiltshire (Wiltshire)

            "E06000030",                         # Swindon Unitary (Wiltshire)

                #2
            # Unitary authority - disregard "E06000024",                         # North Somerset (Somerset)
                #3
            "E07000078",                         # Cheltenham (Gloucestershire)
            "E07000079",                         # Cotswold (Gloucestershire)
            "E07000080",                         # Forest of Dean (Gloucestershire)
            "E07000081",                         # Gloucester (Gloucestershire)
            "E07000082",                         # Stroud (Gloucestershire)
            "E07000083",                         # Tewkesbury (Gloucestershire)

            "E06000025",                         # South Gloucester Unitary (Gloucestershire)

                #4
            "E07000223",                         # Adur (West Sussex)
            "E07000224",                         # Arun (West Sussex)
            "E07000225",                         # Chichester (West Sussex)
            "E07000226",                         # Crawley (West Sussex)
            "E07000227",                         # Horsham (West Sussex)
            "E07000228",                         # Mid Sussex (West Sussex)
            "E07000229"#,                        # Worthing (West Sussex)
#                 #5
#             # Unitary authority - disregard "E06000037",                         # West Berkshire (Berkshire)
#                 #6
#             # Unitary authority - disregard "E06000036",                         # Bracknell Forest (Berkshire)
#                 #7
#             # Unitary authority - disregard "E06000025",                         # South Gloucestershire (Gloucestershire)
#                 #8
#             "E07000218",                         # North Warwickshire (Warwickshire)
#             "E07000219",                         # Nuneaton and Bedworth (Warwickshire)
#             "E07000220",                         # Rugby (Warwickshire)
#             "E07000221",                         # Stratford-on-Avon (Warwickshire)
#             "E07000222",                         # Warwick (Warwickshire)
#                 #9
#             "E07000177",                         # Cherwell (Oxfordshire)
#             "E07000178",                         # Oxford (Oxfordshire)
#             "E07000179",                         # South Oxfordshire (Oxfordshire)
#             "E07000180",                         # Vale of White Horse (Oxfordshire)
#             "E07000181",                         # West Oxfordshire (Oxfordshire)
#                 #10
#             "E07000129",                         # Blaby (Leicestershire)
#             "E07000130",                         # Charnwood (Leicestershire)
#             "E07000131",                         # Harborough (Leicestershire)
#             "E07000132",                         # Hinckley and Bosworth (Leicestershire)
#             "E06000016",                         # Leicester (Leicestershire)
#             "E07000133",                         # Melton (Leicestershire)
#             "E07000134",                         # North West Leicestershire (Leicestershire)
#             "E07000135"                          # Oadby and Wigston (Leicestershire)
 ]

# Load CSV
df_LSOA_MSOA_LAD = pd.read_csv(LSOA_MSOA_LAD_FILEPATH, low_memory=False, encoding="cp1252")
# Check result
print("Initial DataFrame shape:", df_LSOA_MSOA_LAD.shape)

# Keep only required columns
df_LSOA_MSOA_LAD = df_LSOA_MSOA_LAD[keep_cols]

# Filter rows where ladcd is in the specified codes
df_LSOA_MSOA_LAD = df_LSOA_MSOA_LAD[df_LSOA_MSOA_LAD["ladcd"].isin(lad_codes)]

#Distinct list of LSOA codes
valid_lsoa_codes = set(df_LSOA_MSOA_LAD["lsoa21cd"].unique())

# Check result
print("Filtered DataFrame shape:", df_LSOA_MSOA_LAD.shape)

output_path_LML = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_filtered_LSOA_MSOA_LAD.csv"
df_LSOA_MSOA_LAD.to_csv(output_path_LML, index=False)
print(f"Saved filtered LSOA MSOA LAD file to: {output_path_LML}")


# In[10]:


# Ensure that LSOA were fitlered correctly
unique_lad_codes = df_LSOA_MSOA_LAD["ladcd"].unique().tolist()
unique_lad_names = df_LSOA_MSOA_LAD["ladnm"].unique().tolist()

print("Unique LAD names in filtered data:", unique_lad_names)

# Compare with lad_codes list
missing_codes= set(lad_codes) - set(unique_lad_codes)
extra_codes = set(unique_lad_codes) - set(lad_codes)
print("Missing LAD codes from filter:", missing_codes)
print("Unexpected LAD codes in filtered data:", extra_codes)

# Validate row count per LAD code
print("\nRow count per LAD code:")
print(df_LSOA_MSOA_LAD["ladcd"].value_counts())

# Validate LSOA names sample
print("\nSample LSOA names:")
print(df_LSOA_MSOA_LAD["lsoa21nm"].head().tolist())


# ## Load and combine census data

# In[11]:


# Location of files
p_ts067 = CENSUS_DIR / CENSUS_FILENAMES[0]
p_ts044 = CENSUS_DIR / CENSUS_FILENAMES[1]

# Read files
df_ts067 = pd.read_csv(p_ts067, low_memory=False, encoding="cp1252")
df_ts044 = pd.read_csv(p_ts044, low_memory=False, encoding="cp1252")

key_cols = list(set(df_ts067.columns) & set(df_ts044.columns))

print("Common columns:", key_cols)
print("df_ts067 original shape:", df_ts067.shape)
print("df_ts044 original shape:", df_ts044.shape)

# Normalise headers to avoid stray whitespace/BOM;
df_ts067.columns = [c.strip().replace("\ufeff", "") for c in df_ts067.columns]
df_ts044.columns = [c.strip().replace("\ufeff", "") for c in df_ts044.columns]

# Non-key values
nonkey_067 = [c for c in df_ts067.columns if c not in key_cols]
nonkey_044 = [c for c in df_ts044.columns if c not in key_cols]

df_ts067 = df_ts067[key_cols + nonkey_067]
df_ts044 = df_ts044[key_cols + nonkey_044]

# Merge dataframes into one
df_census = pd.merge(
    df_ts067,
    df_ts044,
    on=key_cols,
    how="outer",
    suffixes=("_ts067", "_ts044")
)

print("df_census original shape:", df_census.shape)
print("Key columns:", key_cols)
print("Sample columns:", df_census.columns.tolist()[:12])

print("Sample df_census:")
print(df_census.head())


# ## Filter census to selected LSOAs

# In[12]:


# Filter census to LSOAs
df_census = df_census[df_census["geography code"].isin(valid_lsoa_codes)]

print("df_census shape after filtering:", df_census.shape)

output_path_c = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_census_combined.csv"
df_census.to_csv(output_path_c, index=False)
print(f"Saved merged census file to: {output_path_c}")


# ## Load population

# In[13]:


# Load CSV
df_population = pd.read_csv(POPULATION_FILEPATH, low_memory=False, encoding="cp1252")
print("df_population original shape:", df_population.shape)


# In[14]:


# Filter census to LSOAs
df_population = df_population[df_population["LSOA 2021 Code"].isin(valid_lsoa_codes)]
print("df_population shape after filtering:", df_population.shape)

output_path_pf = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_population_filtered.csv"
df_population.to_csv(output_path_pf, index=False)
print(f"Saved fitlered population file to: {output_path_pf}")


# ## Merge population and census

# In[15]:


df_population.shape


# In[16]:


df_population.columns


# In[17]:


df_census.shape


# In[18]:


df_census.columns


# In[19]:


# Columns to bring from population
all_cols = df_population.columns.tolist()

pop_cols = ['Total 2015', 'Total 2016', 'Total 2017', 'Total 2018',
       'Total 2019', 'Total 2020', 'Total 2021', 'Total 2022']

# Normalise join keys (trim, make string to preserve leading zeros) 
df_population['LSOA 2021 Code'] = (df_population['LSOA 2021 Code'].astype(str).str.strip())
df_census['geography code'] = (df_census['geography code'].astype(str).str.strip())

df_population_agg = df_population[['LSOA 2021 Code'] + [c for c in pop_cols if c in df_population.columns]]

# Left join population onto census 
df_census_with_pop = pd.merge(
    df_census,
    df_population_agg,
    left_on='geography code',
    right_on='LSOA 2021 Code',
    how='left',
    validate='one_to_one'  # change to 'one_to_many' if census has duplicates by geography code
)

# Quick validation 
total_rows = len(df_census)
matched_rows = df_census_with_pop[pop_cols].notna().any(axis=1).sum()
print(f"Rows in df_census: {total_rows}")
print(f"Rows with at least one population value matched: {matched_rows}")
missing_keys = df_census.loc[~df_census['geography code'].isin(df_population_agg['LSOA 2021 Code']), 'geography code'].unique()
print(f"Distinct census geography codes with no population match: {len(missing_keys)}")

# Peek at result
display(df_census_with_pop.head())

output_path_cwp = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_census_with_population.csv"
df_census_with_pop.to_csv(output_path_cwp, index=False)
print(f"Saved census with population file to: {output_path_cwp}")


# ## Alter census data

# In[20]:


# Calculate Under 16 population
df_census_with_pop["Under 16 (2021)"] = (
    df_census_with_pop["Total 2021"]
    - df_census_with_pop["Highest level of qualification: Total: All usual residents aged 16 years and over"]
)

# Replace any negative results (from data issues) with 0
df_census_with_pop["Under 16 (2021)"] = df_census_with_pop["Under 16 (2021)"].clip(lower=0)

# Quick validation checks
rows_missing_inputs = df_census_with_pop[["Total 2021", "Highest level of qualification: Total: All usual residents aged 16 years and over"]].isna().any(axis=1).sum()
rows_negative_before_clip = (df_census_with_pop["Total 2021"] - df_census_with_pop["Highest level of qualification: Total: All usual residents aged 16 years and over"] < 0).sum()

print(f"Computed 'Under 16 (2021)' for {df_census_with_pop.shape[0] - rows_missing_inputs} rows/ {df_census_with_pop.shape[0]} rows.")
print(f"Rows with missing input values: {rows_missing_inputs}")
print(f"Rows that would be negative prior to clipping: {rows_negative_before_clip}")

# Preview
display(df_census_with_pop[["geography code", "Total 2021",
                   "Highest level of qualification: Total: All usual residents aged 16 years and over",
                   "Under 16 (2021)"]].head())

# Qualification per 10k rates:
qual_cols = [
    'Highest level of qualification: Total: All usual residents aged 16 years and over',
    'Highest level of qualification: No qualifications',
    'Highest level of qualification: Level 1 and entry level qualifications',
    'Highest level of qualification: Level 2 qualifications',
    'Highest level of qualification: Apprenticeship',
    'Highest level of qualification: Level 3 qualifications',
    'Highest level of qualification: Level 4 qualifications and above',
    'Highest level of qualification: Other qualifications'
]

base_pop_qual = 'Highest level of qualification: Total: All usual residents aged 16 years and over'

for col in qual_cols[1:]:  # skip the first because it's the base
    rate_col = f"{col} (per 10k - 2021)"
    df_census_with_pop[rate_col] = ((df_census_with_pop[col] / df_census_with_pop[base_pop_qual]) * 10000).round(2)

qual_rate_cols = [f"{col} (per 10k - 2021)" for col in qual_cols[1:]]

print("Added qual rate columns:", qual_rate_cols)
display(df_census_with_pop[['geography code'] + qual_rate_cols].head())

# Accomodation per 10k rates:
# Need to convert households to households per 10k, then use other cols to calculate the per 10k rate.
base_pop_accom = "Total 2021"

accom_cols = [
    'Accommodation type: Total: All households',
    'Accommodation type: Detached', 
    'Accommodation type: Semi-detached',
    'Accommodation type: Terraced',
    'Accommodation type: In a purpose-built block of flats or tenement',
    'Accommodation type: Part of a converted or shared house, including bedsits',
    'Accommodation type: Part of another converted building, for example, former school, church or warehouse',
    'Accommodation type: In a commercial building, for example, in an office building, hotel or over a shop',
    'Accommodation type: A caravan or other mobile or temporary structure']

for col in accom_cols:
    rate_col = f"{col} (per 10k - 2021)"
    df_census_with_pop[rate_col] = ((df_census_with_pop[col] / df_census_with_pop[base_pop_accom]) * 10000).round(2)

accom_rate_cols = [f"{col} (per 10k - 2021)" for col in accom_cols]

print("Added accom rate columns:", accom_rate_cols)
display(df_census_with_pop[['geography code'] + accom_rate_cols].head())

# Drop old versions now we have the rates
cols_to_drop = [
       'Highest level of qualification: Total: All usual residents aged 16 years and over',
       'Highest level of qualification: No qualifications',
       'Highest level of qualification: Level 1 and entry level qualifications',
       'Highest level of qualification: Level 2 qualifications',
       'Highest level of qualification: Apprenticeship',
       'Highest level of qualification: Level 3 qualifications',
       'Highest level of qualification: Level 4 qualifications and above',
       'Highest level of qualification: Other qualifications',
       'Accommodation type: Total: All households',
       'Accommodation type: Detached', 'Accommodation type: Semi-detached',
       'Accommodation type: Terraced',
       'Accommodation type: In a purpose-built block of flats or tenement',
       'Accommodation type: Part of a converted or shared house, including bedsits',
       'Accommodation type: Part of another converted building, for example, former school, church or warehouse',
       'Accommodation type: In a commercial building, for example, in an office building, hotel or over a shop',
       'Accommodation type: A caravan or other mobile or temporary structure',
       'LSOA 2021 Code'
]

print("df_census_with_pop before dropping:", df_census_with_pop.shape)
df_census_with_pop.drop(columns=cols_to_drop, inplace=True)
print("df_census_with_pop after dropping:", df_census_with_pop.shape)

output_path_cwpa = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_census_with_population_altered.csv"
df_census_with_pop.to_csv(output_path_cwpa, index=False)
print(f"Saved census with population file to: {output_path_cwpa}")


# ## Move crime files
# Run once to move all objects

# In[21]:


# # Validate root exists to avoid accidental creation of a wrong path
# if not ROOT.exists():
#     raise FileNotFoundError(
#         f"Root folder not found: {ROOT}\n"
#         "Check the path spelling and that the drive/user exists."
#     )

# street_target = ROOT / STREET_DIRNAME
# outcomes_target = ROOT / OUTCOMES_DIRNAME

# # Create target folders (parents=True avoids WinError 3 if intermediate dirs are missing)
# street_target.mkdir(parents=True, exist_ok=True)
# outcomes_target.mkdir(parents=True, exist_ok=True)

# def resolve_collision(target_dir: Path, filename: str) -> Path:
#     """Return a non-colliding path by appending _n before the extension."""
#     p = target_dir / filename
#     if not p.exists():
#         return p
#     stem = p.stem
#     suffix = p.suffix
#     n = 1
#     while True:
#         candidate = target_dir / f"{stem}_{n}{suffix}"
#         if not candidate.exists():
#             return candidate
#         n += 1

# def copy_grouped_files():
#     copied = {"street": 0, "outcomes": 0, "skipped": 0}
#     # Search all CSVs in subfolders, skip files already in the target folders
#     for path in ROOT.rglob("*.csv"):
#         if street_target in path.parents or outcomes_target in path.parents:
#             continue
#         name_lower = path.name.lower()

#         if name_lower.endswith(STREET_SUFFIX):
#             target_dir = street_target
#             group = "street"
#         elif name_lower.endswith(OUTCOMES_SUFFIX):
#             target_dir = outcomes_target
#             group = "outcomes"
#         else:
#             copied["skipped"] += 1
#             continue

#         dest = resolve_collision(target_dir, path.name)

#         if DRY_RUN:
#             print(f"[DRY RUN] Copy: {path}  ->  {dest}")
#         else:
#             shutil.copy2(path, dest)
#         copied[group] += 1

#     return copied

# stats = copy_grouped_files()
# print("Completed.")
# print(f"Copied street:   {stats['street']}")
# print(f"Copied outcomes: {stats['outcomes']}")
# print(f"Skipped other:   {stats['skipped']}")


# ## Add filename column

# In[22]:


# # Add a "file name" column (base filename without .csv) to every CSV
# # Processes both central folders: street_files and outcomes_files
# # Safe write: creates a temporary file, then replaces the original

# def add_filename_column(csv_path: Path):
#     # Value to write: yyyy-mm-police-authority-name-suffix (no extension)
#     base = csv_path.stem
#     # Read CSV; low_memory=False avoids dtype inference issues on large files
#     df = pd.read_csv(csv_path, low_memory=False)
#     df["file name"] = base
#     tmp = csv_path.with_suffix(".tmp.csv")
#     df.to_csv(tmp, index=False)
#     tmp.replace(csv_path)

# for d in TARGET_DIRS:
#     if not d.exists():
#         raise FileNotFoundError(f"Folder not found: {d}")
#     for csv_path in d.glob("*.csv"):
#         add_filename_column(csv_path)


# ## Create crime dataframes

# In[23]:


# Folder roots (use Path objects) 

# def read_csv_robust(p: Path) -> pd.DataFrame:
#     """
#     Read a CSV with robust defaults and add a 'file name' column if missing.
#     """
#     try:
#         df = pd.read_csv(p, low_memory=False)
#     except UnicodeDecodeError:
#         df = pd.read_csv(p, low_memory=False, encoding="cp1252")
#     if "file name" not in df.columns:
#         df["file name"] = p.stem
#     return df


# In[24]:


def concat_folder_chunks(folder: Path, chunksize: int = 200_000) -> pd.DataFrame:
    """
    Looks for defined folder, safe error if not found.
    Checks for csv files only.
    For each csv, appends file name as new column if not already there.
    Loads all into the dataframe
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        return pd.DataFrame()
    frames = []
    for p in csvs:
        for chunk in pd.read_csv(p, chunksize=chunksize, low_memory=False, encoding="cp1252"):
            if "file name" not in chunk.columns:
                chunk["file name"] = p.stem
            frames.append(chunk)
    return pd.concat(frames, ignore_index=True, sort=False)


# ## Build from CSV files

# In[25]:


get_ipython().run_cell_magic('time', '', 'def load_and_downcast(path):\n    """\n    Normalise column names to avoid hidden whitespace issues.\n    Downcast integers.\n    Downcast floats.\n    """   \n    df = concat_folder_chunks(path)\n    # normalise column names to avoid hidden whitespace issues\n    df.columns = df.columns.str.strip()\n\n    # downcast integers\n    int_cols = df.select_dtypes(include=["int", "int64", "Int64"]).columns\n    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")\n\n    # downcast floats\n    float_cols = df.select_dtypes(include=["float", "float64"]).columns\n    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")\n    return df\n\ndf_street_crime   = load_and_downcast(STREET_DIR)\ndf_outcomes_crime = load_and_downcast(OUTCOMES_DIR)\n\nprint("df_street_crime:", df_street_crime.shape, "from", STREET_DIR)\nprint("df_outcomes_crime:", df_outcomes_crime.shape, "from", OUTCOMES_DIR)\n')


# ## Filter out non-relevant LSOA datafields

# In[26]:


# Filter street to LSOAs
df_street_crime = df_street_crime[df_street_crime["LSOA code"].isin(valid_lsoa_codes)]
print("df_street_crime after filtering:", df_street_crime.shape)

# Filter outcomes to LSOAs
df_outcomes_crime = df_outcomes_crime[df_outcomes_crime["LSOA code"].isin(valid_lsoa_codes)]
print("df_outcomes_crime shape after filtering:", df_outcomes_crime.shape)


# ## Creates fall-backs

# In[27]:


# Fall backs to avoid having to re-do large job if rolling back during dev.
output_path_of = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_outcomes_filtered.csv"
df_outcomes_crime.to_csv(output_path_of, index=False)
print(f"Saved filtered outcomes crime file to: {output_path_of}")

output_path_cf = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_street_filtered.csv"
df_street_crime.to_csv(output_path_cf, index=False)
print(f"Saved filtered street crime file to: {output_path_cf}")


# # Short cut to avoid re-merging all the files

# In[150]:


#Avoid above steps if re-running by using fall-backs
df_street_crime = pd.read_csv(output_path_cf, low_memory=False, encoding="cp1252")
print("df_street_crime after filtering:", df_street_crime.shape)

df_outcomes_crime = pd.read_csv(output_path_of, low_memory=False, encoding="cp1252")
print("df_outcomes_crime shape after filtering:", df_outcomes_crime.shape)


# ## Clean & standardise Crime ID in both datasets

# In[151]:


for df in (df_street_crime, df_outcomes_crime):
    df["Crime ID"] = (
        df["Crime ID"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA})
    )


# ## Move STREET rows with no Crime ID → no‑ref dataframe
# 

# In[152]:


# To add back in later
df_street_no_ref = df_street_crime[df_street_crime["Crime ID"].isna()].copy()
df_street_no_ref["Note"] = "No Crime ID"

df_street_crime = df_street_crime[df_street_crime["Crime ID"].notna()].copy()
print("Street (after removing no-ID):", df_street_crime.shape)


# ## Clean OUTCOMES with no Crime ID or no Outcome type - these were duplicating rows

# In[153]:


# Data cleansing
df_outcomes_crime["Outcome type"] = (
    df_outcomes_crime["Outcome type"]
    .astype("string")
    .str.strip()
    .replace({"": pd.NA})
)

df_outcomes_crime = df_outcomes_crime.dropna(subset=["Crime ID", "Outcome type"]).copy()
print("Outcome (after removing empty):", df_outcomes_crime.shape)


# ## Aggregate multiple outcomes per Crime ID
# 

# In[154]:


get_ipython().run_cell_magic('time', '', '# Removes issue where multiple outcomes bloat stree crime\ndef join_unique(s):\n    vals = pd.unique(s.dropna())\n    return ", ".join(vals)\n\ndf_outcome_agg = (\n    df_outcomes_crime.groupby("Crime ID", as_index=False)\n              .agg({"Outcome type": join_unique})\n)\nprint("Outcome aggregated:", df_outcome_agg.shape)\n')


# ## Remove duplicate Crime IDs from STREET
# 

# In[155]:


# Duplicate data cleansing
mask_dup_by_id = df_street_crime["Crime ID"].duplicated(keep="first")

df_street_dup = df_street_crime[mask_dup_by_id].copy()
df_street_dup["Note"] = "Duplicate Crime ID — removed"

df_street_no_ref = pd.concat([df_street_no_ref, df_street_dup], ignore_index=True)

df_street_crime = df_street_crime[~mask_dup_by_id].copy()
print("Street (deduplicated):", df_street_crime.shape)


# ## Build df_crime_combined via LEFT JOIN

# In[156]:


# Combine dfs via 1:1 left join (avoids inner filtering)
df_crime_combined = pd.merge(
    df_street_crime,
    df_outcome_agg,
    on="Crime ID",
    how="left",
    validate="one_to_one" #this means it breaks when there isn';t a 1:1 - which I want!
)

df_crime_combined["Outcome type"] = df_crime_combined["Outcome type"].fillna("No outcome matched")
print("Combined:", df_crime_combined.shape)


# ## Add back in outcomes that had no matching street crime

# In[157]:


# re-attach outcomes with no street crime with available columns
df_unmatched_outcomes = df_outcome_agg[
    ~df_outcome_agg["Crime ID"].isin(df_street_crime["Crime ID"])
].copy()

df_unmatched_outcomes["Note"] = "Outcome with no matching street record"
df_unmatched_outcomes["Crime type"] = "Unknown — no street record"

df_street_no_ref = pd.concat([df_street_no_ref, df_unmatched_outcomes], ignore_index=True)
print("No-ref dataset:", df_street_no_ref.shape)


# ## Schema check

# In[158]:


# Define the full expected schema is present for df_crime_combined
expected_cols = [
    # core street crime columns
    "Crime ID", "Month", "Reported by", "Falls within", "Longitude", "Latitude",
    "Location", "LSOA code", "LSOA name", "Crime type", "Last outcome category",
    "Context", "file name",
    # outcome column
    "Outcome type",
    # optional note field
    "Note"
]

# Identify missing / unexpected columns
missing_cols = [c for c in expected_cols if c not in df_crime_combined.columns]
extra_cols   = [c for c in df_crime_combined.columns if c not in expected_cols]

print(" SCHEMA VALIDATION ")
print(f"Total columns in df_crime_combined: {len(df_crime_combined.columns)}")

if missing_cols:
    print("\n Missing expected columns:")
    for c in missing_cols:
        print("  -", c)
else:
    print("\n No missing expected columns")

if extra_cols:
    print("\n Unexpected columns found:")
    for c in extra_cols:
        print("  -", c)
else:
    print("\n No unexpected columns")

print("\nFinal column order:")
print(df_crime_combined.columns.tolist())


# ## UNION: df_crime_combined + df_street_no_ref

# In[159]:


# Ensure both DataFrames have the same columns in the same order
union_cols = expected_cols  # from schema block above

# Add any missing columns to df_street_no_ref as empty
for col in union_cols:
    if col not in df_street_no_ref.columns:
        df_street_no_ref[col] = pd.NA

# Reindex to ensure perfect alignment
df_street_no_ref = df_street_no_ref.reindex(columns=union_cols)
df_crime_combined = df_crime_combined.reindex(columns=union_cols)

# Create final union
df_crime_final = pd.concat(
    [df_crime_combined, df_street_no_ref],
    ignore_index=True,
    sort=False
)

print(" FINAL UNION COMPLETE ")
print("df_crime_combined:", df_crime_combined.shape)
print("df_street_no_ref:", df_street_no_ref.shape)
print("df_crime_final:", df_crime_final.shape)

# Save
output_path_final = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_crime_final.csv"
df_crime_final.to_csv(output_path_final, index=False)
print(f"Saved final unified crime dataset to:\n{output_path_final}")


# ## Aggregate df_crime_final by Year (from Month), LSOA code - optional easy add for Crime type, Outcome type, counting rows
# 

# In[160]:


# Aggregate df_crime_final by Year (from Month), LSOA code, Crime type, Outcome type, counting rows

# Ensure required columns exist and are clean
df_crime_agg = df_crime_final.copy()

# Derive Year from 'Month' (format like '2015-01'); coerce errors to NaT then to year
df_crime_agg["Month"] = pd.to_datetime(df_crime_agg["Month"], format="%Y-%m", errors="coerce")
df_crime_agg["Year"] = df_crime_agg["Month"].dt.year.astype("Int64")

# Standardise Outcome type: fill missing with a label to keep them visible in counts
df_crime_agg["Outcome type"] = df_crime_agg["Outcome type"].fillna("No outcome matched")

# Group and count
group_cols = ["Year", "LSOA code"#, "Crime type", "Outcome type"
             ]
df_crime_agg = (
    df_crime_agg.groupby(group_cols, dropna=False)
      .size()
      .reset_index(name="count")
      .sort_values(group_cols)
      .reset_index(drop=True)
)

# Preview and save
print("Aggregated shape:", df_crime_agg.shape)
display(df_crime_agg.head())

output_path = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_crime_aggregated.csv"
df_crime_agg.to_csv(output_path, index=False)
print(f"Saved aggregated file to: {output_path}")


# # Crime rate per 10k

# In[161]:


# Crime dataframe 
crime = df_crime_agg.copy()
crime["Year"] = pd.to_numeric(crime["Year"], errors="coerce").astype("Int64")
census = df_census_with_pop.copy()

# Rename population columns BEFORE loop
rename_map = {
    "Total 2015": "Population 2015",
    "Total 2016": "Population 2016",
    "Total 2017": "Population 2017",
    "Total 2018": "Population 2018",
    "Total 2019": "Population 2019",
    "Total 2020": "Population 2020",
    "Total 2021": "Population 2021",
    "Total 2022": "Population 2022",
}
census = census.rename(columns=rename_map)

years = sorted(int(y) for y in crime["Year"].dropna().unique())
years = [yr for yr in years if f"Population {yr}" in census.columns]  # only years we have a denominator for

out = census.copy()

for yr in years:
    # aggregate counts for that year
    cyr = (
        crime.loc[crime["Year"] == yr]
             .groupby("LSOA code", as_index=False)["count"]
             .sum()
             .rename(columns={"count": f"crime_count_{yr}"})
    )

    # ensure the right frame has both columns (even if empty)
    if cyr.empty:
        cyr = pd.DataFrame({"LSOA code": pd.Series(dtype=str),
                            f"crime_count_{yr}": pd.Series(dtype="Int64")})

    # merge
    out = out.merge(
        cyr,
        left_on="geography code",
        right_on="LSOA code",
        how="left",
        validate="one_to_one"
    ).drop(columns=["LSOA code"])

    #  coalesce any suffix variants to the canonical name 
    base = f"crime_count_{yr}"
    # find any columns that match base or its suffixed variants
    candidates = [c for c in out.columns if c == base or c.startswith(base + "_")]

    if len(candidates) == 0:
        # create if still missing
        out[base] = pd.Series(0, index=out.index, dtype="Int64")
    elif len(candidates) == 1:
        # normalise dtype and fill
        out[base] = (
            pd.to_numeric(out[candidates[0]], errors="coerce")
              .fillna(0)
              .astype("Int64")
        )
        if candidates[0] != base:
            out.drop(columns=[candidates[0]], inplace=True)
    else:
        # two columns like _x/_y; prefer right one if exists, else left
        # heuristic: take max after coercion (since one may be NaN)
        vals = None
        for c in candidates:
            series = pd.to_numeric(out[c], errors="coerce")
            vals = series if vals is None else vals.fillna(series)
        out[base] = vals.fillna(0).astype("Int64")
        # drop all candidate columns except the canonical
        out.drop(columns=[c for c in candidates if c != base], inplace=True)

    # denominator (year-matched pop if present, else 2021)
    pop_col = f"Population {yr}" if f"Population {yr}" in out.columns else "Population 2021"

    # rate per 10k (safe numeric)
    out[f"crime_rate_{yr}_per_10k"] = (
        (out[base] / pd.to_numeric(out[pop_col], errors="coerce")) * 10000
    ).replace([np.inf, -np.inf], np.nan).round(2)

print("Added crime count columns:", [c for c in out.columns if c.startswith("crime_count_")])
print("Added crime rate columns:",  [c for c in out.columns if c.startswith("crime_rate_")])

df_census_with_crime_rates = out

output_path = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_census_with_crime_rates.csv"
out.to_csv(output_path, index=False)
print(f"Saved: {output_path}")


# ## Create dfs tuple
# Used to allow for repeatability of common tasks across multiple dfs, eg EDA.  

# In[162]:


dfs = {
    "LSOA, MSOA, LAD mappings" :df_LSOA_MSOA_LAD,
    "Census merged with population": df_census_with_pop,
    "Crime agg": df_crime_agg,
    "Census and Crime Rates" : df_census_with_crime_rates
}


# ## Check dfs
# 

# In[163]:


print(dfs.items())


# In[164]:


for name, df in dfs.items():
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(num_cols)


# ## EDA on dfs

# ### EDA output

# In[165]:


md_blocks = []

for name, df in dfs.items():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_blocks.append(f"### {name} *EDA Started:* {timestamp}")

    # EDA: shape, memory, describe, info, numeric summary
    md_blocks.append(f"### {name} dataframe\n**Dataframe shape:** {df.shape}")
    b = df.memory_usage(deep=True).sum()
    MiB = b / (1024 * 1024)
    md_blocks.append(f"Memory usage: {MiB:.2f} MiB")
    desc = df.describe().round(1)
    md_blocks.append("**Describe details:**\n" + desc.to_markdown())
    summary = pd.DataFrame({
        "Non-Null Count": df.notnull().sum(),
        "Dtype": df.dtypes,
        "Null Count": df.isnull().sum()
    })
    summary["% Null"] = (summary["Null Count"] / len(df)).round(4)
    md_blocks.append(f"### {name} dataframe Set — Info\n" + summary.to_markdown())

    # Numeric summary
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:  # True if num_cols is empty
        md_blocks.append(f"### {name} dataframe does not contain numeric fields, numeric sumamry skipped")
    else:
        desc_num = df[num_cols].describe().T
        md_blocks.append(f"### {name} dataframe\nNumeric summary:\n" + desc_num.to_markdown())

    # Head/tail preview
    head = df.head(5).astype(object).where(~df.head(5).isna(), None)
    tail = df.tail(5).astype(object).where(~df.tail(5).isna(), None)

    md_blocks.append(f"### {name} dataframe \n**Head details:**\n{head.to_markdown()}\n**Tail details:**\n{tail.to_markdown()}")

    # Missing values
    na_counts = df.isna().sum()
    total_rows = len(df)
    missing_count = df.isna().any(axis=1).sum()
    missing_summary = (
        pd.DataFrame({"Missing": na_counts, "Dtype": df.dtypes})
        .assign(PercentMissing=lambda d: (d["Missing"] / total_rows).round(4))
        .query("Missing > 0")
        .sort_values("Missing", ascending=False)
    )
    md_blocks.append(
        f"### {name} dataframe \n**Missing values count:** <br>{missing_count}\n"
        f"**Missing values per column (with dtypes):**\n{missing_summary.to_markdown()}"
    )

    # Duplicates
    nowformat = datetime.now().strftime("%Y%m%d_%H%M%S_")

    dup_mask = df.duplicated(keep=False)
    dup_count = dup_mask.sum()
    dup_unique_count = df[dup_mask].drop_duplicates().shape[0]
    dup_per_record = dup_count / dup_unique_count if dup_count != 0 else None
    md_blocks.append(
        f"### {name} dataframe \n**Duplicate Summary:**<br>"
        f"- Duplicate row count: {dup_count}<br>"
        f"- Unique duplicate row count: {dup_unique_count}<br>"
        f"- Duplicates per record: {dup_per_record}"
    )

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
display(Markdown(f"**EDA on each df**  \nCompleted at: {timestamp}"))


# In[166]:


display(Markdown('\n\n'.join(md_blocks)))

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
display(Markdown(f"**EDA on each df**  \nCompleted at: {timestamp}"))


# ## Missing Values
# NA summary. Per‑column NA count and dtype.

# In[167]:


# Missing values repeated...I didn't beleive it!
mod_all = df_census_with_crime_rates.copy()

summary = pd.DataFrame({
    "Non-Null Count": mod_all.notnull().sum(),
    "Dtype": mod_all.dtypes,
    "Null Count": mod_all.isnull().sum()
})

summary["% Null"] = (summary["Null Count"] / len(mod_all)).round(4)

total_missing_rows = mod_all.isnull().any(axis=1).sum()
df_cwcr_missing_summary = summary[summary["Null Count"] > 0].sort_values("Null Count", ascending=False)

display(Markdown(f"**Missing values count:** <br>{total_missing_rows}"))
display(Markdown("**Missing values per column (with dtypes):**"))
display(df_cwcr_missing_summary)  

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
display(Markdown(f"**Data quality - missing values**  \nCompleted at: {timestamp}"))


# In[168]:


print("Shape before dropping rows with missing values:", mod_all.shape)

# Drop rows with missing values
mod_all = mod_all.dropna(axis=0, how="any").reset_index(drop=True)

print("Shape after dropping rows with missing values:", mod_all.shape)


# In[169]:


# Outlier investigation & Winsdorisation

df_out = mod_all.copy()   # work on a copy

# Compute z-scores across numeric features
numeric_cols = df_out.select_dtypes(include=[np.number]).columns

Z = df_out[numeric_cols].apply(zscore, nan_policy="omit")

# Max |z| per row — good indicator of multivariate outliers
df_out["max_abs_z"] = Z.abs().max(axis=1)

# Investigate outliers
outliers = df_out[df_out["max_abs_z"] > 3]   # threshold = 3 SD

print(f"Detected {len(outliers)} multivariate outliers (>3 SD).")
display(outliers.head())

# Histogram of outlier severity (theme-aligned)
plt.figure(figsize=(7, 4))
sns.histplot(df_out["max_abs_z"], bins=40)
plt.axvline(3, color=sns.color_palette()[3], ls="--", label="Threshold = 3")
plt.title("Outlier score distribution (max |z| per row)")
plt.legend()
plt.tight_layout()
plt.show()

# Winsorise extreme values (suppression)
def winsorise_series(s, z_thresh=3):
    z = zscore(s, nan_policy="omit")
    upper = s[z < z_thresh].max()
    lower = s[z < z_thresh].min()
    return s.clip(lower=lower, upper=upper)

# Apply winsorisation only to numeric columns
df_out[numeric_cols] = df_out[numeric_cols].apply(winsorise_series)

print("Applied winsorisation to numeric columns (capped extreme values).")

mod_all = df_out.drop(columns=["max_abs_z"])


# ## NumPy workaround	
# In NumPy ≥ 2.0, some internal attributes changed, and certain libraries still expect np.VisibleDeprecationWarning to exist.
# This code checks if the attribute is missing and then assigns it from numpy.exceptions.VisibleDeprecationWarning.
# If the import fails (older NumPy versions already have it), the except block does nothing.

# In[170]:


try:
    # NumPy ≥ 2.0: provide the missing attribute for libraries expecting it
    from numpy.exceptions import VisibleDeprecationWarning as _VDW
    np.VisibleDeprecationWarning = _VDW
except Exception:
    pass  # On NumPy < 2.0 the attribute already exists

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
display(Markdown(f"**NumPy workaround**  \nCompleted at: {timestamp}"))


# ## Sweetviz	EDA report	
# Generate HTML and notebook report.  Runs very slow on big data!

# In[171]:


# Make Sweetviz safe copy and prepare for use
numeric_fill=None
cat_missing_label="Missing"

#Uses samping for speed when using large model
df_sv = mod_all

# Replace pandas.NA everywhere with np.nan (safe for numeric dtypes)
df_sv = df_sv.replace({pd.NA: np.nan})

# Identify dtype groups
num_cols  = df_sv.select_dtypes(include=["number"]).columns.tolist()
bool_cols = df_sv.select_dtypes(include=["boolean"]).columns.tolist()
# Treat plain strings, objects, and categoricals as "text-like"
text_cols = df_sv.select_dtypes(include=["string", "object", "category"]).columns.tolist()

# Numerics: median impute (avoids skew vs fill with 0)
if num_cols:
    imputer_num = SimpleImputer(strategy="median")
    df_sv[num_cols] = imputer_num.fit_transform(df_sv[num_cols])

# Booleans: fill NA as False and cast to plain bool
if bool_cols:
    df_sv[bool_cols] = df_sv[bool_cols].fillna(False).astype(bool)

# Text: fill NA with sentinel and keep as object/string
if text_cols:
    df_sv[text_cols] = df_sv[text_cols].fillna("Missing")

# Unify string dtype if needed
for c in text_cols:
    # keep 'object' to play nicely with encoders that expect object
    df_sv[c] = df_sv[c].astype("object")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
display(Markdown(f"**Prepare for Sweetviz**  \nCompleted at: {timestamp}"))    


# In[172]:


# Run Sweetviz (pairwise can be heavy; turn off if needed)
import sweetviz as sv
# report = sv.analyze(df_sv, pairwise_analysis='off')
report = sv.analyze(df_sv)  # default pairwise_analysis='auto'
nowformat = datetime.now().strftime("%Y%m%d_%H%M%S_")
report.show_html(f"sweetviz_EDA_{nowformat}crime.html")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
display(Markdown(f"**Sweetviz EDA run**  \nCompleted at: {timestamp}"))


# # Scale - standardise continuous features (z-score) before regression

# In[173]:


#  Feature selection (single source of truth) 
def select_features(mod_all: pd.DataFrame) -> list[str]:
    feats = [c for c in mod_all.columns if "(per 10k - 2021)" in c]
    feats += [c for c in mod_all.columns if c.startswith("Population")]
    feats += [c for c in mod_all.columns if c.startswith("Under 16")]
    feats = [c for c in feats if not c.startswith("crime_rate_")]
    feats = [c for c in feats if not c.startswith("crime_count_")]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(mod_all[c])]
    return sorted(set(feats))

feature_cols = select_features(mod_all)

#  Prune collinearity BEFORE scaling or modelling 
def prune_collinearity(mod_all: pd.DataFrame, cols: list[str]) -> list[str]:
    X = mod_all[cols].astype(float)
    drop = []

    # Accommodation: drop total and one subcategory (avoid exact linear dependence)
    accom = [c for c in X.columns if "Accommodation type:" in c and "(per 10k - 2021)" in c]
    drop += [c for c in accom if "Total: All households" in c]
    for candidate in [
        "A caravan or other mobile or temporary structure",
        "In a commercial building",
        "Part of another converted building",
    ]:
        match = [c for c in accom if candidate in c]
        if match:
            drop += match[:1]
            break

    # Qualifications: drop any 'Total:' rate variable (use k-1 categories)
    qual = [c for c in X.columns if "Highest level of qualification:" in c and "(per 10k - 2021)" in c]
    drop += [c for c in qual if ": Total:" in c]

    # Constants and exact duplicates
    drop += [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    dup_mask = X.round(12).T.duplicated()
    drop += list(X.columns[dup_mask])

    keep = [c for c in cols if c not in set(drop)]
    return keep

FEATURE_COLS = prune_collinearity(mod_all, feature_cols)

#  VIF prune to a threshold (e.g., 10) 
def vif_iterative(X: pd.DataFrame, thresh=10.0, max_iter=20) -> list[str]:
    cols = list(X.columns)
    for _ in range(max_iter):
        Xn = X[cols].dropna()
        if Xn.shape[1] <= 1:
            break
        vifs = [variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1])]
        worst_idx = int(np.nanargmax(vifs))
        worst_vif = vifs[worst_idx]
        if not np.isfinite(worst_vif) or worst_vif < thresh:
            break
        cols.remove(cols[worst_idx])
    return cols

FEATURE_COLS = vif_iterative(mod_all[FEATURE_COLS].astype(float), thresh=10.0)

#  Now scale / model safely 
scaler = StandardScaler()
X_cont = mod_all[FEATURE_COLS].astype(float)
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_cont),
    columns=FEATURE_COLS,
    index=mod_all.index
)


# ## Correlation structure (redundancy / multicollinearity)
# which features move together; candidates to drop or combine; inputs for VIF.

# In[174]:


corr = pd.DataFrame(X_scaled, columns=X_scaled.columns).corr(method="pearson")

plt.figure(figsize=(14, 10))
ax = sns.heatmap(corr, center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.6, "pad": 0.02})
ax.set_xticklabels(corr.columns, fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9, rotation=0)
plt.title("Feature Correlation (Pearson)")
plt.tight_layout()
plt.show()

# Top absolute correlations (feature pairs)
tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairs = (
    tri.stack()
       .reindex(tri.stack().abs().sort_values(ascending=False).index)
       .rename("corr")
)
display(pairs.head(20))


# ## PCA: structure + dimensionality
# how many dimensions carry most variance; whether data lie on a lower‑dimensional manifold; quick 2D view.

# In[175]:


pca = PCA(n_components=min(20, X_scaled.shape[1])).fit(X_scaled)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)

fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].bar(range(1, len(evr)+1), evr)
ax[0].set_title("PCA: variance explained per component")
ax[0].set_xlabel("PC")
ax[0].set_ylabel("Explained variance ratio")

ax[1].plot(range(1, len(cum)+1), cum, marker="o")
ax[1].axhline(0.90, color="red", ls="--", label="90%")
ax[1].set_title("PCA: cumulative variance")
ax[1].set_xlabel("PC")
ax[1].set_ylabel("Cumulative explained variance")
ax[1].legend()
plt.tight_layout()
plt.show()

# 2D PCA projection (optionally colour by a target if present in `mod`)
Z = PCA(n_components=2).fit_transform(X_scaled)
pc_df = pd.DataFrame(Z, columns=["PC1","PC2"], index=mod_all.index)

plt.figure(figsize=(7,6))
if "crime_rate_2021_per_10k" in mod_all.columns:
    sc = plt.scatter(pc_df["PC1"], pc_df["PC2"],
                     c=mod_all["crime_rate_2021_per_10k"], s=12)
    plt.colorbar(sc, label="crime_rate_2021_per_10k")
else:
    plt.scatter(pc_df["PC1"], pc_df["PC2"], s=12, alpha=0.7)
plt.title("PCA (2D projection)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.show()


# ## Outliers in z‑score space
# rows that are extreme on any standardised feature (potential leverage points).

# In[176]:


# Xz = pd.DataFrame(X_scaled, columns=feature_cols, index=mod_all.index)
Xz = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=mod_all.index)
keep = Xz.columns[Xz.nunique(dropna=True) > 1]

Xz = Xz[keep]
corr = Xz.corr()

row_max_abs_z = Xz.abs().max(axis=1)
outliers = Xz[row_max_abs_z > 3]   # 3-sigma rule, adjust if needed
print(f"Outlier rows (> |z| 3): {outliers.shape[0]} out of {Xz.shape[0]}")
display(outliers.head())

# Quick distribution of max |z| per row
plt.figure(figsize=(7,4))
sns.histplot(row_max_abs_z, bins=50)
plt.axvline(3, color="red", ls="--", label="|z|=3")
plt.title("Max absolute z-score per row")
plt.legend(); plt.tight_layout(); plt.show()


# ## Quick multicollinearity screen (VIF on unscaled or scaled—both acceptable)
# features with unstable coefficients risk.

# In[177]:


def compute_vif_safe(X_df, max_corr_drop=True, verbose=True):
    """
    Compute VIF robustly:
      1) Keep only numeric columns
      2) Drop columns with all-NA or constant values
      3) Drop exact duplicate columns
      4) Optionally drop perfectly correlated columns (|r| == 1 within tolerance)
      5) Add constant and compute VIF for remaining features
    Returns: (vif_table, kept_columns, dropped_columns)
    """
    dropped = []

    # numeric only
    X = X_df.select_dtypes(include=[np.number]).copy()

    # drop all-NA and constant columns
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if const_cols:
        X.drop(columns=const_cols, inplace=True)
        dropped += [(c, "constant_or_allNA") for c in const_cols]

    # drop exact duplicate columns
    # (transpose, mark duplicates, keep first)
    if X.shape[1] > 1:
        dup_mask = X.T.duplicated()
        dup_cols = X.columns[dup_mask].tolist()
        if dup_cols:
            X.drop(columns=dup_cols, inplace=True)
            dropped += [(c, "duplicate_column") for c in dup_cols]

    # drop perfectly correlated columns (|r| == 1 within tol)
    if max_corr_drop and X.shape[1] > 1:
        corr = X.corr().abs()
        # upper triangle without diagonal
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # use a tight tolerance to catch numeric identity after scaling
        perfect_pairs = np.where(np.isclose(upper.values, 1.0, atol=1e-12))
        cols_to_drop = set()
        for i, j in zip(*perfect_pairs):
            # keep the first; drop the later column
            col_j = upper.columns[j]
            if col_j not in cols_to_drop:
                cols_to_drop.add(col_j)
        if cols_to_drop:
            X.drop(columns=list(cols_to_drop), inplace=True)
            dropped += [(c, "perfect_correlation") for c in cols_to_drop]

    # drop remaining columns with any NaN (VIF needs complete rows)
    X_clean = X.dropna(axis=0, how="any")
    # If dropping rows causes empty frame, bail out
    if X_clean.empty or X_clean.shape[1] == 0:
        return (pd.DataFrame(columns=["feature", "VIF"]), X.columns.tolist(), dropped)

    # add constant for VIF regression (required by statsmodels formula)
    X_exog = sm.add_constant(X_clean, has_constant="add")

    # compute VIF per predictor (skip the constant)
    vifs = []
    features = [c for c in X_exog.columns if c != "const"]

    # Ignore FP warnings here; handled exact 1.0 above.
    old_settings = np.seterr(all="ignore")
    try:
        for i, col in enumerate(X_exog.columns):
            if col == "const":
                continue
            idx = i  # position in exog matrix
            v = variance_inflation_factor(X_exog.values, idx)
            # map negative/inf/NaN to large number for readability
            if not np.isfinite(v) or v < 0:
                v = np.inf
            vifs.append((col, float(v)))
    finally:
        np.seterr(**old_settings)

    vif_table = pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False).reset_index(drop=True)

    if verbose:
        print("Dropped columns (reason):", dropped)
    return vif_table, features, dropped


# In[178]:


# VIF as table
vif_table, kept, dropped = compute_vif_safe(X_cont, max_corr_drop=True, verbose=True)
display(vif_table.head(20))
print("Kept:", kept)
print("Dropped:", dropped[:10])  # show first few dropped with reasons


# In[179]:


# Tabular highlight (top offenders only)
top_vif = vif_table.copy()
top_vif["VIF_display"] = top_vif["VIF"].replace([np.inf, -np.inf], np.nan).round(2)
top_vif = top_vif.sort_values("VIF", ascending=False)

display(top_vif.head(20))


# In[180]:


# Clean VIF values (map inf to a cap for plotting)
vif_plot = vif_table.copy()
vif_plot["VIF_plot"] = vif_plot["VIF"].replace([np.inf, -np.inf], np.nan)
cap = np.nanpercentile(vif_plot["VIF_plot"], 95) if np.isfinite(vif_plot["VIF_plot"]).any() else 100.0
vif_plot["VIF_plot"] = vif_plot["VIF_plot"].fillna(cap)  # cap infinities to 95th percentile (or 100)

# Sort descending
vif_plot = vif_plot.sort_values("VIF_plot", ascending=True)

plt.figure(figsize=(10, max(4, 0.35 * len(vif_plot))))
bars = plt.barh(vif_plot["feature"], vif_plot["VIF_plot"], color="#4c78a8")

# Threshold lines (common guidance)
plt.axvline(5, color="orange", ls="--", lw=1.5, label="VIF=5")
plt.axvline(10, color="red", ls="--", lw=1.5, label="VIF=10")

plt.title("Variance Inflation Factor (VIF) by feature")
plt.xlabel("VIF (capped for ∞)")
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.,
)
plt.show()

# Interpretation

# Features to scrutinise: VIF ≥ 10 (strong), 5–10 (moderate).
# Consider dropping/combining those, or using regularised models (Ridge/Lasso).


# In[181]:


# Tabular highlight (top offenders only)
top_vif = vif_table.copy()
top_vif["VIF_display"] = top_vif["VIF"].replace([np.inf, -np.inf], np.nan).round(2)
top_vif = top_vif.sort_values("VIF", ascending=False)

display(top_vif.head(20))


# ## Hierarchical clustering of features
# Groups of near‑duplicate features.

# In[182]:


# Convert square -> condensed
dist_sq = ((1 - corr.fillna(0)) / 2.0)**0.5   # square matrix
dist_condensed = squareform(dist_sq.values, checks=False)  # 1-D

Z = linkage(dist_condensed, method="average")

plt.figure(figsize=(12, 4))
dendrogram(Z, labels=corr.columns, leaf_rotation=90)
plt.title("Feature Dendrogram (correlation distance)")
plt.show()


# In[183]:


#Quickest path to a clustered heatmap without manual linkage:
sns.clustermap(corr, center=0, vmin=-1, vmax=1,
               method="average", metric="correlation", figsize=(12, 10))
plt.show()


# ## Baseline modelling (per Year or pooled)

# In[184]:


y = mod_all["crime_rate_2021_per_10k"].astype(float)
X = sm.add_constant(X_scaled)  # add intercept
ols = sm.OLS(y, X, missing="drop").fit()
print(ols.summary())   # includes R^2, p-values, ANOVA (via .anova_lm if nested models)


# In[185]:


# Model diagnostics:
# Residual plots: residuals vs fitted, QQ plot.
# Assumptions: homoscedasticity (Breusch–Pagan), normality (Shapiro/QQ), independence (Durbin–Watson).
dw = sm.stats.durbin_watson(ols.resid)

print(dw)


# ## Error metrics vs hold-out
# Use grouped split to avoid leakage (e.g., hold out a set of LSOAs):

# In[186]:


# configuration 
TARGET_COL = "crime_rate_2021_per_10k"

# Columns needed for this spec
model_cols = ["geography code", TARGET_COL] + FEATURE_COLS

# Start from the canonical table and drop rows with any NA in target or features
mod_base = df_census_with_crime_rates[model_cols].dropna(axis=0, how="any").copy()

# Attach LAD groups to the SAME rows used for modelling
lad_map = df_LSOA_MSOA_LAD[["lsoa21cd", "ladcd"]].drop_duplicates()
mod_2021 = (
    mod_base
    .merge(lad_map, left_on="geography code", right_on="lsoa21cd", how="left")
    .drop(columns=["lsoa21cd"])
    .dropna(subset=["ladcd"])           # drop rows without a group
    .reset_index(drop=True)
)

# Define X, y, groups from the SAME dataframe
X_all  = mod_2021[FEATURE_COLS].astype(float)
y_all  = mod_2021[TARGET_COL].astype(float)
groups = mod_2021["ladcd"].astype(str)

# Sanity checks
assert len(X_all) == len(y_all) == len(groups)


# In[187]:


#Quick diagnostics: rank, duplicates, perfect correlation
X_df = mod_all[FEATURE_COLS].astype(float)

# Matrix rank vs number of columns
rank = np.linalg.matrix_rank(X_df.values)
print(f"X shape: {X_df.shape}, matrix rank: {rank}")

# Constant / all-NA columns
const_cols = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 1]
print("Constant/all-NA columns:", const_cols)

# Exact duplicates (after rounding to stabilise floating noise)
dup_mask = X_df.round(12).T.duplicated()
dup_cols = X_df.columns[dup_mask].tolist()
print("Exact duplicate columns:", dup_cols)

# Perfect correlations (|r|==1 within tolerance)
corr = X_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
perfect_pairs = [(i,j) for i in upper.columns for j in upper.columns
                 if upper.loc[i,j] >= 1 - 1e-12]
print("Perfectly correlated pairs (if any):", perfect_pairs[:10])


# In[188]:


drop_cols = []
#Drop obvious collinearity sources (recommended rules for data)
# Drop accommodation total and one subcategory (choose a small/rare one to drop)
accom_cols = [c for c in X_df.columns if "Accommodation type:" in c and "(per 10k - 2021)" in c]
drop_cols += [c for c in accom_cols if "Total: All households" in c]
# drop one subcategory to avoid exact sum-to-total trap
for candidate in [
    "A caravan or other mobile or temporary structure",
    "In a commercial building",
    "Part of another converted building",
]:
    match = [c for c in accom_cols if candidate in c]
    if match:
        drop_cols += match[:1]
        break

# Ensure qualification block uses k-1 (skip total and one level if needed)
qual_cols = [c for c in X_df.columns if "Highest level of qualification:" in c and "(per 10k - 2021)" in c]
# (likely already excluded the base; if not, drop the 'Total:' rate)
drop_cols += [c for c in qual_cols if ": Total:" in c]  # safety

# Drop constants/duplicates found above
drop_cols += const_cols + dup_cols

drop_cols = sorted(set([c for c in drop_cols if c in X_df.columns]))
print("Dropping columns (rule-based):", drop_cols)

FEATURE_COLS = [c for c in FEATURE_COLS if c not in drop_cols]
X_df = X_df.drop(columns=drop_cols)


# In[189]:


#Ensure full rank after pruning (and optionally prune by VIF)
rank = np.linalg.matrix_rank(X_df.values)
print(f"Post-prune X shape: {X_df.shape}, rank: {rank}")

# Optional: iterative VIF prune to threshold (e.g., 10)
def vif_iterative(X, thresh=10.0, max_iter=20):
    cols = list(X.columns)
    for _ in range(max_iter):
        Xn = X[cols].dropna()
        if Xn.shape[1] <= 1: break
        vifs = [variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1])]
        worst_idx = int(np.nanargmax(vifs))
        worst_vif = vifs[worst_idx]
        if worst_vif < thresh or not np.isfinite(worst_vif): break
        drop = cols[worst_idx]
        cols.remove(drop)
        print(f"Dropping by VIF: {drop} (VIF={worst_vif:.2f})")
    return cols

FEATURE_COLS = vif_iterative(X_df, thresh=10.0)
X_df = X_df[FEATURE_COLS]

# IMPORTANT: rebuild the global feature matrix AFTER pruning
X_all = mod_all[FEATURE_COLS].copy()

print("Kept features:", FEATURE_COLS)


# # Predictive modelling

# ## Test/train split (80/20)

# In[190]:


# Downcast numerics on X_all for memory efficiency (floats only)
num_cols = X_all.select_dtypes(include=["float64", "float32", "float16", "int64", "int32", "int16"]).columns
X_all[num_cols] = X_all[num_cols].apply(pd.to_numeric, downcast="float")

# Grouped train/test split to prevent LAD leakage (80/20)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_all, y_all, groups))

X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

# Markdown summary (shapes + group integrity)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
md = [
    "**Downcast metrics and grouped train/test split**",
    f"Completed at: {timestamp}",
    "",
    f"**Train X:** {X_train.shape}  <br>"
    f"**Test X:** {X_test.shape}  <br>"
    f"**Train y:** {y_train.shape}  <br>"
    f"**Test y:** {y_test.shape}",
    "",
    "**Group integrity (LAD codes):**",
    f"- Unique LADs (train): {groups_train.nunique()}",
    f"- Unique LADs (test): {groups_test.nunique()}",
    f"- Overlap LADs between train/test: {len(set(groups_train) & set(groups_test))} (should be 0 for leakage-free grouped split)"
]
display(Markdown("\n\n".join(md)))

# Simple visual: bar chart of sample counts and a pie of group share
fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=130)

# Bar: row counts
axes[0].bar(["Train", "Test"], [len(X_train), len(X_test)], color=["#4c78a8", "#f58518"])
axes[0].set_title("Rows per split")
axes[0].set_ylabel("Rows")

# Pie: share of LAD groups by count of rows (train vs test)
sizes = [len(X_train), len(X_test)]
axes[1].pie(sizes, labels=["Train", "Test"], autopct="%1.1f%%", colors=["#4c78a8", "#f58518"], startangle=90)
axes[1].set_title("Split proportion")

plt.tight_layout()
plt.show()

# Quick sanity on target distribution parity
fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=130)
sns.kdeplot(y_train, label="Train", ax=ax)
sns.kdeplot(y_test, label="Test", ax=ax, linestyle="--")
ax.set_title("Target distribution: Train vs Test")
ax.legend()
plt.show()


# In[191]:


# GroupKFold by LAD with in-fold scaling (no leakage) on train
# creates K different train/test folds
# each fold uses different subsets of LADs
# folds exist only inside the loop
# after the loop, have no X_train/X_test stored anywhere
# the “test set” changes every fold
# the outputs kept are RMSE/MAE/MAPE per fold, not a split
n_splits = min(5, groups_train.nunique())     # ensure enough LADs for the split count
gkf = GroupKFold(n_splits=n_splits)

rmse, mae, mape = [], [], []
for tr, te in gkf.split(X_train, y_train, groups_train):
    # fit scaler ONLY on train, then transform both
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train.iloc[tr]), columns=FEATURE_COLS, index=X_train.index[tr])
    X_te = pd.DataFrame(scaler.transform(X_train.iloc[te]),     columns=FEATURE_COLS, index=X_train.index[te])

    y_tr, y_te = y_train.iloc[tr], y_train.iloc[te]

    m = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
    pred = m.predict(sm.add_constant(X_te))

    rmse.append(np.sqrt(mean_squared_error(y_te, pred)))
    mae.append(mean_absolute_error(y_te, pred))
    mape.append((np.abs((y_te - pred) / y_te.replace(0, np.nan))).mean())

print("CV Results (GroupKFold, grouped by LAD):")
print(f"RMSE: mean={np.mean(rmse):.3f}, std={np.std(rmse):.3f}")
print(f"MAE:  mean={np.mean(mae):.3f}, std={np.std(mae):.3f}")
print(f"MAPE: mean={np.mean(mape):.3f}, std={np.std(mape):.3f}")

metrics_df = pd.DataFrame({"Fold": range(1, len(rmse)+1), "RMSE": rmse, "MAE": mae, "MAPE": mape})
display(metrics_df)

output_path_ols_group_cv_metrics_df = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ols_group_cv_metrics_df.csv"
metrics_df.to_csv(output_path_ols_group_cv_metrics_df, index=False)
print(f"Saved OLS group CV metrics output to: {output_path_ols_group_cv_metrics_df}")

plt.figure(figsize=(10,5))
sns.boxplot(data=metrics_df[["RMSE","MAE","MAPE"]])
plt.title("Cross-validated model errors (Grouped by LAD)")
plt.ylabel("Error")
plt.show()


# # EDA on TRAIN

# In[192]:


# Fit final model on TRAIN only
scaler_final = StandardScaler().fit(X_train)
Xtr = scaler_final.transform(X_train)
Xte = scaler_final.transform(X_test)

final_pred_model = sm.OLS(y_train, sm.add_constant(Xtr)).fit()

print(final_pred_model.summary())


# In[193]:


#Evaluate the final model on the TEST set
pred_test = final_pred_model.predict(sm.add_constant(Xte))

rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
mae_test  = mean_absolute_error(y_test, pred_test)
mape_test = np.mean(np.abs((y_test - pred_test) / y_test.replace(0, np.nan)))

print("FINAL MODEL – Test set performance:")
print("RMSE:", rmse_test)
print("MAE :", mae_test)
print("MAPE:", mape_test)


# In[194]:


# Add baseline Poisson/NB for counts with offset to compare specs:
y_cnt = mod_all["crime_count_2021"].astype(float)
X_nb  = sm.add_constant(mod_all[FEATURE_COLS].astype(float))
offset = np.log(mod_all["Population 2021"].clip(lower=1))
nb = sm.GLM(y_cnt, X_nb, family=sm.families.NegativeBinomial(alpha=1.0), offset=offset).fit()
print(nb.summary())


# # Interpretive / Scientific modelling.

# In[195]:


# OLS
ols_classic = sm.OLS(y, sm.add_constant(X_scaled)).fit()
ols_hc3    = sm.OLS(y, sm.add_constant(X_scaled)).fit(cov_type="HC3")
print(ols_hc3.summary())

final_model = ols_classic
# Provides:
# Coefficient estimates
# p‑values
# overall regression fit (R², F‑statistic)
# confidence intervals

# This is not the model to evaluate, but the model to interpret.


# ## Coefficient forest with 95% CI
# A compact view of effect sizes and uncertainty. 

# ## Coefficient forest with 95% CI
# A compact view of effect sizes and uncertainty. 

# ## Coefficient forest with 95% CI
# A compact view of effect sizes and uncertainty. 

# In[196]:


params = ols_hc3.params
conf   = ols_hc3.conf_int()
coef   = pd.DataFrame({"coef": params, "lo": conf[0], "hi": conf[1]})
coef = coef.drop(index="const").sort_values("coef")
plt.figure(figsize=(7, 0.35*len(coef)))
plt.hlines(y=coef.index, xmin=coef["lo"], xmax=coef["hi"], color="#4c78a8")
plt.plot(coef["coef"], coef.index, "o", color="#1f77b4")
plt.axvline(0, color="grey", lw=1)
plt.title("OLS (HC3) coefficients with 95% CI")
# plt.tight_layout();
plt.show()


# ## Partial Regression Plot (Added‑Variable Plot)
# A Partial Regression Plot—also called an Added‑Variable Plot—shows the unique relationship between a predictor and the outcome after controlling for all other predictors in the model

# In[197]:


# choose top-K by absolute HC3 coefficient (or by ElasticNet if prefered)
K = 7
top_feats = (
    ols_hc3.params.drop("const").abs().sort_values(ascending=False).index[:K].tolist()
    # alternatively: pd.Index(FEATURE_EN)[:K].tolist()
)

# build a reduced model with the top-K features
X_top = sm.add_constant(mod_all[top_feats].astype(float))
ols_top = sm.OLS(y, X_top).fit()

# draw larger partial-regress grid
fig = sm.graphics.plot_partregress_grid(
    ols_top, fig=plt.figure(figsize=(18, 12), dpi=150)
)
for ax in fig.axes:
    ax.tick_params(labelsize=10)
plt.tight_layout()
plt.show()


# ## Scale-location plot
# Detect non-constant variance.

# In[198]:


resid = ols_hc3.resid
fitted = ols_hc3.fittedvalues
plt.figure(figsize=(6,4))
sns.scatterplot(x=fitted, y=np.sqrt(np.abs(resid)), s=12)
plt.title("Scale-location"); plt.xlabel("Fitted"); plt.ylabel("Residual")
plt.axhline(np.sqrt(np.median(np.abs(resid))), color="red", ls="--"); plt.show()


# ## Influence and Cook’s distance
# Find influential LSOAs.

# In[199]:


infl = ols_classic.get_influence()
cooks = infl.cooks_distance[0]
plt.figure(figsize=(7,4))
plt.stem(range(len(cooks)), cooks, basefmt=" ")
plt.title("Cook's distance by observation"); plt.xlabel("Observation"); plt.ylabel("Cook's D")
plt.show()


# ## Look at coefficient importance
# Sort by magnitude (absolute value):

# In[200]:


coef_df = pd.DataFrame({
    "feature": ["Intercept"] + list(X_scaled.columns),
    "coef": final_model.params.values
}).assign(abs_coef=lambda d: d["coef"].abs()).sort_values("abs_coef", ascending=False)

display(coef_df.head(20))


# ## Sanity check residuals
# Residuals vs Fitted, Residual Distribution, Quantile–Quantile Plot

# In[201]:


resid = final_model.resid
fitted = final_model.fittedvalues

plt.figure(figsize=(6,4))
sns.scatterplot(x=fitted, y=resid, s=10)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Fitted")
plt.show()

sns.histplot(resid, kde=True)
plt.title("Residual distribution")
plt.show()

sm.qqplot(resid, line='45')
plt.show()


# The LinAlgError: Singular matrix in ML_Lag means design matrix X is not full‑rank. In practice this is caused by one or more of:
# 
# a constant or duplicate column
# perfectly (or near‑perfectly) collinear features (e.g., category sets that sum to a total)
# more features than effective observations after filtering

# In[202]:


# # Reuse pruned FEATURE_COLS from step 1
# TARGET_COL = "crime_rate_2021_per_10k"

# # Align data to weights id order
# order = list(w.id_order)
# mod_sub = mod.set_index(mod[ID_COL].astype(str).str.strip()).reindex(order)

# # Drop rows where target or any feature is missing
# cols_needed = [TARGET_COL] + FEATURE_COLS
# valid_mask = ~mod_sub[cols_needed].isna().any(axis=1)
# mod_sub = mod_sub.loc[valid_mask]

# # Subset weights to valid ids (manual subset function already have)
# w = subset_weights_manual(w, mod_sub.index.tolist())

# # Final rank check before fitting
# print("Final X shape:", mod_sub[FEATURE_COLS].shape,
#       "rank:", np.linalg.matrix_rank(mod_sub[FEATURE_COLS].astype(float).values))

# y = mod_sub[TARGET_COL].astype(float).values.reshape(-1, 1)
# X = mod_sub[FEATURE_COLS].astype(float).values


# In[203]:


# #    Use consolidated table (you set earlier):
# mod = df_census_with_crime_rates.copy()

# # Target and predictors
# TARGET_COL   = "crime_rate_2021_per_10k"     # <-- or another year’s rate
# FEATURE_COLS = feature_cols                   # <-- from earlier feature selection


# # Spatial dependence (SAR) to test spillover hypothesis
# Requirement: an LSOA spatial weights matrix W (adjacency).
# Data: ONS LSOA 2021 boundaries (GeoJSON/Shapefile).
# Method: queen contiguity or k-nearest neighbours.

# Notes and guidance
# 
# 
# Weights: Queen contiguity captures polygon adjacency; KNN fallback ensures that “islands” (LSOAs with no neighbours) still get neighbours (k=6 is typical; adjust if needed).
# 
# 
# Alignment: SAR requires y, X, and W to be in identical observation order. The code reindexes mod to w.id_order and sub‑selects W to match dropped rows.
# 
# 
# Target: I used crime_rate_2021_per_10k. Can swap in another year’s rate or a count with an offset (for Poisson/NB).
# 
# 
# Features: FEATURE_COLS should be numeric predictors (e.g., per‑10k socio‑economic indicators). Exclude identifiers and target columns.
# 
# 
# Diagnostics:
# 
# LM tests (spat_diag=True) are printed in the PySAL summary.
# Moran’s I of residuals checks remaining spatial autocorrelation. If still significant, consider SEM (spatial error), SAC (lag+error), or different features.
# 
# 
# 
# Model comparison: Use AIC/BIC and residual Moran’s I to compare OLS vs SAR. For predictive comparison, OOS CV for SAR is non‑trivial because predictions depend on neighbours; if needed, we can implement blocked spatial CV that refits SAR per fold and evaluates in‑fold fit or uses a reduced WWW on the held‑out graph.

# ## Spatial Analysis

# In[204]:


# Pre GDF setup
mod_modelling = df_census_with_crime_rates.copy()  # ensure 'mod' exists before filtering boundaries

BOUNDARY_PATH = r"C:\Users\cxputte\Crime\Source\LSOA_2021_Boundaries\england_lsoa_2021.shp"
CODE_FIELD    = "lsoa21cd"  # check this matches the shapefile's field name exactly

gdf = gpd.read_file(BOUNDARY_PATH)
gdf[CODE_FIELD] = gdf[CODE_FIELD].astype(str).str.strip()

print("Boundary columns:", gdf.columns.tolist())
print("Boundary CRS:", gdf.crs)
print(gdf.head(2))


# In[205]:


#Filter boundaries to modelling IDs and set CRS
ID_COL = "geography code" if "geography code" in mod_modelling.columns else "LSOA code"
needed_ids = mod_modelling[ID_COL].astype(str).str.strip().unique()

gdf = gdf[gdf[CODE_FIELD].isin(needed_ids)].copy()
try:
    gdf = gdf.to_crs(4326)                  # WGS84 system used
except Exception:
    pass

# ids in boundary order (used later to align data)
ids = gdf[CODE_FIELD].astype(str).str.strip().tolist()
print("Shapefile loaded. Number of LSOAs after filtering:", len(gdf))


# In[206]:


#    Use consolidated table
mod_spatial = df_census_with_crime_rates.copy()

# Target and predictors
TARGET_COL   = "crime_rate_2021_per_10k"     # <-- or another year’s rate
FEATURE_COLS = feature_cols                   # <-- from earlier feature selection


# In[207]:


# def build_weights_for_ids(gdf, code_field, ids, k_fallback=6):
#     """
#     Rebuild Queen weights for the given id list (order preserved).
#     Adds KNN fallback for islands, then row-standardises.
#     """
#     # subset gdf to requested ids, preserving order
#     sub = gdf.set_index(code_field).loc[ids].reset_index()
#     wq = Queen.from_dataframe(sub, ids=ids)
#     islands = [k for k, v in wq.cardinalities.items() if v == 0]
#     if islands:
#         wkn = KNN.from_dataframe(sub, k=k_fallback, ids=ids)
#         # manual union (see Fix B) or use the union function below
#         w = union_weights(wq, wkn)
#     else:
#         w = wq
#     w.transform = "r"
#     return w


# In[208]:


# Keep only LSOAs present in modelling table to speed-up
ids_needed = set(mod_spatial[ID_COL].astype(str).str.strip().unique())
gdf = gdf[gdf[CODE_FIELD].astype(str).str.strip().isin(ids_needed)].copy()
gdf = gdf.to_crs(4326) if gdf.crs is not None else gdf  # WSG84

# Build queen + knn
wq  = weights.Queen.from_dataframe(gdf, ids=ids)
wkn = weights.KNN.from_dataframe(gdf, k=6, ids=ids)

def union_weights(w1: weights.W, w2: weights.W) -> weights.W:
    if list(w1.id_order) != list(w2.id_order):
        # align w2 to w1 order
        w2 = weights.W({i: w2.neighbors.get(i, []) for i in w1.id_order}, id_order=w1.id_order)
    id_order = list(w1.id_order)
    neigh = {i: sorted(set(w1.neighbors.get(i, [])) | set(w2.neighbors.get(i, []))) for i in id_order}
    return weights.W(neigh, id_order=id_order)

islands = [k for k, v in wq.cardinalities.items() if v == 0]
w = union_weights(wq, wkn) if islands else wq
w.transform = "r"

# Verify weights cardinalities after union:
deg = pd.Series(w.cardinalities)
assert (deg > 0).all(), "Islands remain after KNN fallback"

# Start with candidate features
X_df = mod_spatial[FEATURE_COLS].astype(float)

# Drop constants and duplicates
const_cols = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 1]
dup_mask   = X_df.round(12).T.duplicated()
dup_cols   = X_df.columns[dup_mask].tolist()

# Drop accommodation total and one small subcategory
accom_cols = [c for c in X_df.columns if "Accommodation type:" in c and "(per 10k - 2021)" in c]
drop_cols  = [c for c in accom_cols if "Total: All households" in c]
for candidate in [
    "A caravan or other mobile or temporary structure",
    "In a commercial building",
    "Part of another converted building",
]:
    match = [c for c in accom_cols if candidate in c]
    if match:
        drop_cols += match[:1]
        break

# Qualification safety: drop any 'Total:' rate variable if present
qual_cols = [c for c in X_df.columns if "Highest level of qualification:" in c and "(per 10k - 2021)" in c]
drop_cols += [c for c in qual_cols if ": Total:" in c]

# Add constants and duplicates to drop list
drop_cols += const_cols + dup_cols
drop_cols  = sorted(set(drop_cols))
FEATURE_COLS = [c for c in FEATURE_COLS if c not in drop_cols]
X_df = X_df.drop(columns=[c for c in drop_cols if c in X_df.columns])

# Optional VIF pruning
def vif_iterative(X, thresh=10.0, max_iter=20):
    cols = list(X.columns)
    for _ in range(max_iter):
        Xn = X[cols].dropna()
        if Xn.shape[1] <= 1:
            break
        vifs = [variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1])]
        worst_idx = int(np.nanargmax(vifs))
        worst_vif = vifs[worst_idx]
        if worst_vif < thresh or not np.isfinite(worst_vif):
            break
        drop = cols[worst_idx]
        cols.remove(drop)
        print(f"Dropping by VIF: {drop} (VIF={worst_vif:.2f})")
    return cols

FEATURE_COLS = vif_iterative(X_df, thresh=10.0)
print("Kept features:", FEATURE_COLS)

# Align data to weights id order
order = list(w.id_order)
mod_sub = mod_spatial.set_index(mod_spatial[ID_COL].astype(str).str.strip()).reindex(order)

# Drop rows where target or any feature is missing
cols_needed = [TARGET_COL] + FEATURE_COLS
valid_mask = ~mod_sub[cols_needed].isna().any(axis=1)
mod_sub = mod_sub.loc[valid_mask]

# Subset weights to valid ids (manual subset)
def subset_weights_manual(w_obj: weights.W, keep_ids: list) -> weights.W:
    keep = set(keep_ids)
    id_order = list(keep_ids)
    neigh = {i: [j for j in w_obj.neighbors.get(i, []) if j in keep] for i in id_order}
    w_sub = weights.W(neigh, id_order=id_order)
    w_sub.transform = "r"
    return w_sub

order = list(w.id_order)
mod_sub = mod_spatial.set_index(mod_spatial[ID_COL].astype(str).str.strip()).reindex(order)
valid_mask = ~mod_sub[[TARGET_COL] + FEATURE_COLS].isna().any(axis=1)
mod_sub = mod_sub.loc[valid_mask]
w = subset_weights_manual(w, mod_sub.index.tolist())

# Confirm alignment before fitting:
assert list(mod_sub.index) == list(w.id_order)

# Build y and X matrices
# Ensure X is full rank at this point
print("Final X shape:", mod_sub[FEATURE_COLS].shape,
      "rank:", np.linalg.matrix_rank(mod_sub[FEATURE_COLS].astype(float).values))

y = mod_sub[TARGET_COL].astype(float).values.reshape(-1, 1)
X = mod_sub[FEATURE_COLS].astype(float).values  # no constant; ML_Lag adds it when constant=True

# OLS baseline
ols = sm.OLS(y, sm.add_constant(X)).fit()
print("\n=== OLS baseline ===")
print(ols.summary())

# SAR lag
sar = ML_Lag(y, X, w,
             name_y=TARGET_COL,
             name_x=FEATURE_COLS,
             spat_diag=True,
             name_w="W_QueenKNN",
             constant=True)
print("\n=== SAR (ML_Lag) summary ===")
print(sar.summary)

print("\n=== SAR (ML_Lag) summary ===")
# The PySAL models expose a text summary via `summary` or `summary` property
try:
    print(sar.summary)
except Exception:
    # Fallback: print key pieces
    betas = np.asarray(sar.betas).ravel()
    names = ["Intercept"] + FEATURE_COLS
    print("Coefficients:")
    for n, b in zip(names, betas):
        print(f"{n:30s} {b: .6f}")
    print(f"AIC: {getattr(sar, 'aic', np.nan)}   BIC: {getattr(sar, 'schwarz', np.nan)}   pseudo-R2: {getattr(sar, 'pr2', np.nan)}")

#  Residual diagnostics: Moran's I on SAR residuals 
# Common residual attribute names in spreg: `u` (residuals), `resid`, or y - predy
resid = getattr(sar, "u", None)
if resid is None:
    resid = getattr(sar, "resid", None)
if resid is None:
    predy = getattr(sar, "predy", None)
    resid = (y.ravel() - predy.ravel()) if predy is not None else (y.ravel() - ols.fittedvalues.ravel())

mi = Moran(resid, w)
print("\n=== Residual spatial autocorrelation (Moran's I) ===")
print(f"I = {mi.I: .5f},  p_norm = {mi.p_norm: .5f}")
if mi.p_norm < 0.05:
    print("Residuals still show significant spatial autocorrelation (p<0.05). Consider alternative specs (e.g., SEM, SAC) or different features.")
else:
    print("No significant residual spatial autocorrelation detected at 5% level.")

#  Model comparison quick view 
def safe(val, default=np.nan):
    try:
        return float(val)
    except Exception:
        return default

sar_aic = safe(getattr(sar, "aic", np.nan))
sar_bic = safe(getattr(sar, "schwarz", np.nan))
sar_pr2 = safe(getattr(sar, "pr2", np.nan))

print("\n=== Model comparison (in-sample) ===")
print(f"OLS:  AIC={ols.aic:.2f},  BIC={ols.bic:.2f},  R²={ols.rsquared:.3f}")
print(f"SAR:  AIC={sar_aic:.2f},  BIC={sar_bic:.2f},  pseudo-R²={sar_pr2:.3f}")


sem = ML_Error(y, X, w, name_y=TARGET_COL, name_x=FEATURE_COLS, spat_diag=True, constant=True)
print("\n=== SEM (ML_Error) summary ===")
print(sem.summary)

#  Coefficient table 
try:
    betas = np.asarray(sar.betas).ravel()
    names = ["Intercept"] + FEATURE_COLS
    coef_df = pd.DataFrame({"feature": names, "coef": betas})
    display(coef_df.sort_values("coef", key=np.abs, ascending=False).head(20))
except Exception:
    pass

output_path_features_used = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_features_used.csv"
pd.Series(FEATURE_COLS, name="feature").to_csv(output_path_features_used, index=False)
print(f"Saved: {output_path_features_used}")
# pd.Series(FEATURE_COLS, name="feature").to_csv(outpath("features_used.csv"), index=False)

output_path_coef_df = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_coef_df.csv"
coef_df.to_csv(output_path_coef_df, index=False)
print(f"Saved: {output_path_coef_df}")
# coef_df.to_csv(outpath("ols_coefficients.csv"), index=False)



# In[209]:


# alignment (safe against duplicate ID column)
ID_COL = "geography code" if "geography code" in mod_spatial.columns else "LSOA code"
order = list(w.id_order)

mod_sub = (
    mod_spatial.copy()
    .assign(**{ID_COL: mod_spatial[ID_COL].astype(str).str.strip()})
    .set_index(ID_COL)
    .reindex(order)
)

output_path_mod_sub_all = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mod_sub_all.csv"
# mod_sub.to_csv(output_path_mod_sub_all, index=False)
mod_sub.reset_index(names=CODE_FIELD).to_csv(output_path_mod_sub_all, index=False)
print(f"Saved: {output_path_mod_sub_all}")

# if the ID remains as a column, drop it to avoid duplicates
if ID_COL in mod_sub.columns:
    mod_sub = mod_sub.drop(columns=[ID_COL])

# Ensure the GeoDataFrame index carries the correct name
gdf_mod = gdf.set_index(CODE_FIELD).loc[mod_sub.index].copy()
gdf_mod.index.name = CODE_FIELD  
# Attach measures
gdf_mod["crime_rate_2021_per_10k"] = mod_sub["crime_rate_2021_per_10k"].values
gdf_mod["fitted_ols"] = ols.fittedvalues
gdf_mod["resid_ols"]  = ols.resid

# Visual
ax = gdf_mod.plot(column="fitted_ols", legend=True, figsize=(8,8), linewidth=0.1, edgecolor="#f0f0f0")   # very light grey

ax.set_axis_off()
plt.title("OLS fitted crime rate 2021 per 10k"); plt.show()

# Export .geojson
lsoa_subset_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lsoa_subset.geojson"
output_path_lsoa_subset = OUTPUTS_PATH / lsoa_subset_filename
gdf_mod.to_file(output_path_lsoa_subset, driver="GeoJSON")
print(f"Saved: {output_path_lsoa_subset}")

# # Export .shp
# lsoa_subset_filename_s = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lsoasub.shp"
# output_path_lsoa_subset_s = OUTPUTS_PATH / lsoa_subset_filename_s
# gdf_mod.to_file(output_path_lsoa_subset_s, driver="ESRI Shapefile")
# print(f"Saved: {output_path_lsoa_subset_s}")

# # model table aligned (index -> column named CODE_FIELD)
# output_path_model_table_aligned = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_model_table_aligned.csv"

# gdf_mod.reset_index()[[CODE_FIELD, "crime_rate_2021_per_10k", "fitted_ols", "resid_ols"]].to_csv(
#     output_path_model_table_aligned, index=False
# )

# print(f"Saved: {output_path_model_table_aligned}")

 # Robust column selection after reset_index
cols_needed = [CODE_FIELD, "crime_rate_2021_per_10k", "fitted_ols", "resid_ols"]
df_out = gdf_mod.reset_index()

# If for any reason the index column comes out as 'index', rename it to CODE_FIELD
if CODE_FIELD not in df_out.columns and "index" in df_out.columns:
    df_out = df_out.rename(columns={"index": CODE_FIELD})

missing = [c for c in cols_needed if c not in df_out.columns]
if missing:
    raise KeyError(f"Missing columns in export: {missing}")

output_path_model_table_aligned = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_model_table_aligned.csv"
df_out[cols_needed].to_csv(output_path_model_table_aligned, index=False)
print(f"Saved: {output_path_model_table_aligned}")

output_path_mod_sub = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mod_sub.csv"
mod_sub.to_csv(output_path_mod_sub, index=False)
print(f"Saved: {output_path_mod_sub}")


# In[210]:


mod_sub.head()


# In[211]:


# gdf_mod = gdf.set_index(CODE_FIELD).loc[mod_sub.index]
# gdf_mod["fitted_ols"] = ols.fittedvalues
# gdf_mod["resid_ols"]  = ols.resid
# ax = gdf_mod.plot(column="fitted_ols", cmap="viridis", legend=True, figsize=(8,8))
# ax.set_axis_off()
# plt.title("OLS fitted crime rate 2021 per 10k"); plt.show()

# output_path_lsoa_subset = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lsoa_subset.geojson"
# gdf_mod.to_file(output_path_lsoa_subset, driver="GeoJSON")
# print(f"Saved: {output_path_lsoa_subset}")

# output_path_model_table_aligned = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_model_table_aligned.csv"
# gdf_mod.reset_index()[[CODE_FIELD, "crime_rate_2021_per_10k", "fitted_ols", "resid_ols"]].to_csv(output_path_model_table_aligned, index=False)
# # mod_sub.reset_index().to_csv(output_path_model_table_aligned, index=False)
# print(f"Saved: {output_path_model_table_aligned}")


# In[212]:


# Export Diagnostic results from spatial models, including Moran’s I values and LM tests

output_path_spatial_metrics = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_spatial_metrics.csv"
pd.DataFrame(
    {"stat": ["moran_I", "moran_p_norm", "ols_aic", "ols_bic", "ols_r2",
              "sar_aic", "sar_bic", "sar_pseudo_r2"],
     "value": [mi.I, mi.p_norm, ols.aic, ols.bic, ols.rsquared,
               getattr(sar, "aic", np.nan), getattr(sar, "schwarz", np.nan), getattr(sar, "pr2", np.nan)]}
).to_csv(output_path_spatial_metrics, index=False)
print(f"Saved: {output_path_spatial_metrics}")


# ## Spatial diagnostics visuals
# Moran scatter of OLS residuals to complement numeric Moran’s I.
# Choropleth of residuals quantiles for spatial patterns.

# In[213]:


# res = ols_classic.resid.values
# mi = Moran(res, w)
# fig, _ = moran_scatterplot(mi)
# plt.show()

# gdf_mod["resid_q"] = pd.qcut(gdf_mod["resid_ols"], 5, labels=False, duplicates="drop")
# ax = gdf_mod.plot(column="resid_q", legend=True, figsize=(8,8))
# ax.set_axis_off(); plt.title("OLS residuals - quintiles"); plt.show()


# In[214]:


res = ols_classic.resid.values
mi = Moran(res, w)

res_lag = weights.lag_spatial(w, res)

# Layout
fig = plt.figure(figsize=(12, 10), dpi=150)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.25)

# PANEL 1
ax1 = fig.add_subplot(gs[0])

sns.regplot(
    x=res,
    y=res_lag,
    scatter_kws=dict(alpha=0.45, s=28, color="#4c78a8", edgecolor="none"),
    line_kws=dict(color="#c44536", linewidth=2),
    ax=ax1
)

ax1.axhline(0, color="grey", linewidth=0.8)
ax1.axvline(0, color="grey", linewidth=0.8)
ax1.set_title(f"Moran Scatterplot (I = {mi.I:.3f})", fontsize=13)
ax1.set_xlabel("Residual", fontsize=11)
ax1.set_ylabel("Spatial lag of residual", fontsize=11)

# PANEL 2 — residual map
gdf_mod["resid_q"] = pd.qcut(res, 5, labels=False, duplicates="drop")
ax2 = fig.add_subplot(gs[1])

gdf_mod.plot(
    column="resid_q",
    linewidth=0.1,
    edgecolor="#f0f0f0",
    legend=True,
    legend_kwds=dict(
        label="Residual quintile",
        orientation="vertical",
        shrink=0.7,
        pad=0.02
    ),
    ax=ax2
)

ax2.set_axis_off()
ax2.set_title("OLS residuals — quintiles", fontsize=13)

plt.show()


# ## Predicted vs observed with calibration band

# In[215]:


plt.figure(figsize=(6,6))
plt.scatter(ols_hc3.fittedvalues, y, s=10, alpha=0.6)
lims = [min(y.min(), ols_hc3.fittedvalues.min()), max(y.max(), ols_hc3.fittedvalues.max())]
plt.plot(lims, lims, "r--"); plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title("Calibration"); plt.show()


# ## Regularised linear baseline
# Helps with multicollinearity and yields a sparse, readable model.

# In[216]:


enet = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, random_state=42))
])

enet.fit(mod_spatial[FEATURE_COLS].astype(float), y.ravel())
coef_en = pd.Series(enet.named_steps["model"].coef_, index=FEATURE_COLS).sort_values()
selected = coef_en[coef_en != 0].index.tolist()

coef_sorted = coef_en.sort_values()
print(coef_sorted)

selected = coef_en[coef_en != 0].index.tolist()
print(selected)

y_pred = enet.predict(mod_spatial[FEATURE_COLS].astype(float))
resid = y - y_pred

rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
print('RMSE: ', rmse, 'MAE: ', mae)


plt.scatter(y_pred, y, s=10, alpha=0.5)
plt.xlabel("Predicted")
plt.ylabel("Observed")
plt.title("ElasticNet – predicted vs observed")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.show()

FEATURE_EN = selected   # ElasticNet-selected predictors

#OLS with HC3 robust SE
X_en = sm.add_constant(mod_spatial[FEATURE_EN].astype(float))
ols_en = sm.OLS(y, X_en).fit(cov_type="HC3")
print(ols_en.summary())

#SAR with reduced features

# ensure ndarray with correct shape
y_sar = np.asarray(y, dtype=float).reshape(-1, 1)


# predictors matrix

# use sparse LU for stable logdet; optionally tune epsilon
X_sar = mod_spatial[FEATURE_EN].astype(float).values
sar_en = ML_Lag(y_sar, X_sar, w,
                method="LU",            
                epsilon=1e-6,           # optional: solver tolerance
                name_y="crime_rate_2021_per_10k",
                name_x=FEATURE_EN,
                constant=True)

# X_sar = mod_spatial[FEATURE_EN].astype(float).values

# sar_en = ML_Lag(y_sar, X_sar, w, name_y="crime_rate_2021_per_10k", name_x=FEATURE_EN, constant=True)

# X_sar = mod_spatial[FEATURE_EN].astype(float).values
# sar_en = ML_Lag(y.values.reshape(-1,1), X_sar, w,
#                 name_y="crime_rate_2021", name_x=FEATURE_EN,
#                 constant=True)
print(sar_en.summary)

#simple coefficient plot
coef_sorted.plot(kind="barh", figsize=(6,0.3*len(coef_sorted)))
plt.title("ElasticNet coefficients")
plt.show()

(coef_en * 100000).plot(kind="barh", figsize=(8, 0.3*len(coef_en)))
plt.title("ElasticNet coefficients × 100000")
plt.show()


output_path_elasticnet_sf = OUTPUTS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_elasticnet_selected_features.csv"
pd.Series(selected).to_csv(output_path_elasticnet_sf, index=False)
print(f"Saved: {output_path_elasticnet_sf}")


# ## Panel formulation across years
# Exploit all years by stacking counts and using an offset.

# In[217]:


rate_cols  = [c for c in mod_spatial.columns if c.startswith("crime_rate_") and c.endswith("_per_10k")]
count_cols = [c for c in mod_spatial.columns if c.startswith("crime_count_")]
pop_cols   = [c for c in mod_spatial.columns if c.startswith("Population ")]

long = []
for yr in sorted(int(c.split("_")[2]) for c in rate_cols):
    tmp = mod_spatial[["geography code"] + FEATURE_COLS + [f"crime_count_{yr}", f"Population {yr}"]].copy()
    tmp["year"] = yr
    long.append(tmp.rename(columns={f"crime_count_{yr}":"count", f"Population {yr}":"pop"}))
panel = pd.concat(long, axis=0, ignore_index=True).dropna()

X_nb = sm.add_constant(panel[FEATURE_COLS].astype(float))
y_nb = panel["count"].astype(float)
offset = np.log(panel["pop"].clip(lower=1))
nb_panel = sm.GLM(y_nb, X_nb, family=sm.families.NegativeBinomial(alpha=1.0), offset=offset).fit()
print(nb_panel.summary())


# In[218]:


# Open Power BI Desktop
# Go to Options → Preview Features
# Turn on Shape Map Visual.
# Restart Power BI.
# Add the Shape Map visual.
# Import the TopoJSON (lsoa_2021_subset.topojson).
# _lsoa_subset.geojson
# Add CSV:
# _model_table_aligned.csv
# lsoa21cd, crime_rate_2021_per_10k, fitted_ols, resid_ols


# Create a relationship:

# TopoJSON field: lsoa21cd
# CSV field: lsoa21cd


# Use crime_rate_2021_per_10k or fitted_ols as Colour.


# Important notes
# 
# 
# Prediction on a held-out graph uses only within‑fold neighbours. If there are cross‑border links between train and test areas in the full graph, they are intentionally excluded here to avoid leakage. That is the standard approach for blocked spatial CV.
# 
# 
# Grouping choice affects difficulty. Group by LAD if available to hold out whole areas. If LAD is not present, the k‑means centroids fallback creates spatially coherent folds.
# 
# 
# Convergence. If SAR fails on some folds due to singularity, reduce features or apply regularisation in the feature selection stage (e.g., drop highly collinear indicators using VIF results).
# 
# 
# Targets. The example uses a rate target. For crime counts, consider Poisson or NB with offsets rather than SAR lag OLS.

# # Plan

# In[219]:


# 5) Ethics, bias, and privacy

# Measurement bias: policing intensity varies by area → include proxy controls (e.g., number of officers per LAD if available) or caution in interpretation.
# Statistical parity: compare residuals or rates by LAD to ensure no systematic over/under-estimation for specific areas.
# Privacy: all data are aggregate; continue to exclude personal identifiers; retain only LSOA-level outputs.
# GDPR lawful basis: document public task / research basis; store outputs in timestamped OUTPUTS_PATH; avoid row-level joins to any sensitive sources.

# 6) Limitations and multicollinearity handling

# Imputed census variables: flag rows where imputation occurred; sensitivity analysis excluding those.
# VIF: drop or combine highly collinear features; consider PCA as an alternative (report explained variance).
# Robustness: try robust SE (HC3) in OLS:
# # ols_robust = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")


