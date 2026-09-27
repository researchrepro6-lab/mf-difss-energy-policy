# Raw data sources (snapshot used for the manuscript)

All files in this folder are unmodified downloads, except `ccpi_2023_2024_ranks.csv`,
which is a transcription (see below). `run_all.py` reads only these files.
`download_data.py` in the repository root re-downloads the public files. Upstream
providers revise their data, so a fresh download may differ from this snapshot; the
SHA-256 checksums below identify the exact files analysed.

| File | Provider / dataset | Access URL | Retrieved | Licence | Used for |
|---|---|---|---|---|---|
| `WB_RISE.csv` | World Bank, Regulatory Indicators for Sustainable Energy (RISE), Data360 database `WB_RISE` | https://data360files.worldbank.org/data360-data/data/WB_RISE/WB_RISE.csv | 2026-09-23 | CC BY 4.0 | Policy criterion e1 (indicator `WB_RISE_RE_ALL`, reference years 2015, 2017, 2019, 2021, 2023) and its five sub-indicators (`WB_RISE_RE_ELEC`, `_GOV`, `_HE_CO`, `_LVL_PLYNG_FLD`, `_TRNS`) for hesitancy |
| `owid-energy-data.csv`, `owid-energy-codebook.csv` | Our World in Data energy dataset (Ember Yearly Electricity Data; Energy Institute Statistical Review) | https://github.com/owid/energy-data | 2026-09-23 | CC BY 4.0 | Renewable electricity share e2 (`renewables_share_elec`, % of total generation including nuclear) |
| `API_EN.GHG.CO2.PC.CE.AR5_DS2_en_excel_v2_33405.xls` | World Bank WDI, EN.GHG.CO2.PC.CE.AR5 (CO2 emissions excl. LULUCF, t CO2e per capita) | https://data.worldbank.org/indicator/EN.GHG.CO2.PC.CE.AR5 | 2026-08-31 | CC BY 4.0 | CO2 criterion e3 |
| `rise-2016-database-for-download-1.xlsx` | World Bank / ESMAP, RISE 2016 downloadable database (2015 regulatory information) | https://rise.esmap.org | 2026-08-26 | CC BY 4.0 | Sensitivity only: legacy single-observation policy criterion (renewable-energy pillar score) |
| `Renewable_Energy.csv` | IMF Climate Change Indicators Dashboard, Renewable Energy dataset (IRENA-based) | https://climatedata.imf.org | 2026-08-31 | IMF terms of use (attribution) | Sensitivity only: alternative renewable share RE/(RE+fossil); the file contains no nuclear or geothermal generation |
| `ccpi_2023_2024_ranks.csv` | Climate Change Performance Index 2023 and 2024 rankings (Burck et al.; Germanwatch, NewClimate Institute, CAN) | Transcribed from the tabulation of official results in the Wikipedia article "Climate Change Performance Index" (revision of 2026-09-23), which cites https://ccpi.org/wp-content/uploads/CCPI-2023-Results-3.pdf and CCPI-2024-Results.pdf | 2026-09-23 | CCPI: non-commercial use; table derived from CC BY-SA text | External validation only |

**Note:** the CCPI ranks are a transcription of a secondary tabulation of the official
results. Check them against the official CCPI 2023 and 2024 results before reusing them.

## SHA-256 checksums

```
700a382b80000a66b568816c1114cc1ef38a911b634632f37d066b20caf1cf4d  API_EN.GHG.CO2.PC.CE.AR5_DS2_en_excel_v2_33405.xls
3fe23068f687ee0fbf7b3aadae43b40d9584db0e3bdc716093d554e474318ed5  Renewable_Energy.csv
05d5dd89edd08b30746b17a6ff83758ccb10cf9d5cacf0e136ee7fac62cd497f  WB_RISE.csv
04f458fd50546c70bea5b2a331450ec00bba3901de3655db8d82a175bafd9b2a  ccpi_2023_2024_ranks.csv
3cc9b7db0d921496e2988568ce3aee5ed41f50431dd234a0663b5f0a4b2e32bb  owid-energy-codebook.csv
266f2e2baad7975351bc9bb4aa061d22b1da9fe4c47d51d2ac6071e01e171f76  owid-energy-data.csv
0ed85a3a6db05c131ecb8e12b2d0d8730d2437de9e6c3642b6f58e727c0797aa  rise-2016-database-for-download-1.xlsx
```

## Notes on definitions

* **RISE reference years.** Data360 distributes the RISE scores as an annual 2010-2023
  series, while RISE is published biennially (editions 2016, 2018, 2020, 2022 and 2024; the
  2024 edition required data collection to be completed by 31 December 2023). Each edition
  reports the policies in place at the end of the preceding year, so the analysis keeps only
  the reference years 2015, 2017, 2019, 2021 and 2023, and the policy criterion keeps its sparse, irregular cadence relative to the
  annual criteria. The Data360 scores follow the current RISE methodology and are not
  numerically comparable with the legacy RISE 2016 workbook; see the sensitivity analysis.
* **Renewable share.** Renewable generation as defined by Ember/OWID (hydropower, wind,
  solar, bioenergy and other renewables such as geothermal and marine) divided by total
  generation from all sources, including nuclear. Pumped-storage conventions follow Ember.
* **Country matching.** Countries are matched on ISO3 codes, and aggregates are excluded
  using the World Bank country metadata. Name-based matching is used only for the legacy
  RISE 2016 workbook and the CCPI table; see `data/crosswalk/`.
