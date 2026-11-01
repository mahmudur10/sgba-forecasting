# Data Sources

This directory holds the raw and processed PV datasets used in the paper.
Both sources are **publicly available**; no proprietary data is included.

## 1. DKASC (Alice Springs, Australia)

**Provider:** Desert Knowledge Australia Solar Centre
**URL:** https://dkasolarcentre.com.au/download
**Climate:** Desert (arid), high irradiance, low average cloud cover
**Sites used:** 3 subsystems

| Site ID | Technology | Rated Power | Mounting | Installed |
|---------|-----------|-------------|----------|-----------|
| Site 7  | CdTe thin-film | 7.0 kW | Fixed-tilt | 2008 |
| Site 10 | mono-Si         | 5.8 kW | Fixed-tilt | 2009 |
| Site 2  | mono-Si         | 26.5 kW | Dual-axis tracker | 2010 |

**Data period:** 2020–2023 (4 years, ~97.6% coverage)
**Raw resolution:** 5 minutes
**Channels:** AC power (kW), ambient temperature (°C), relative humidity (%),
global horizontal irradiance (W/m²), diffuse horizontal irradiance (W/m²)

### Download instructions

1. Visit https://dkasolarcentre.com.au/download
2. Select sites 7, 10, 2 (Alice Springs)
3. Download CSV files for 2020-01-01 through 2023-12-31
4. Place files in `data/raw/` as:
   - `dkasc_site7_raw.csv`
   - `dkasc_site10_raw.csv`
   - `dkasc_site2_raw.csv`

## 2. PVGIS (Cross-climate validation)

**Provider:** European Commission Joint Research Centre — PVGIS
**URL:** https://joint-research-centre.ec.europa.eu/pvgis-photovoltaic-geographical-information-system_en
**Data type:** Typical Meteorological Year (TMY) — synthetic data covering
2016–2020 averaged into a representative year.

| Site | Latitude | Longitude | Climate |
|------|----------|-----------|---------|
| Singapore   | 1.3521° N | 103.8198° E | Tropical |
| Oslo        | 59.9139° N | 10.7522° E | Cold-continental |
| Madrid      | 40.4168° N | −3.7038° E | Mediterranean |

### Download instructions

Use the `pvlib` Python interface:

```python
import pvlib

# Example: download Singapore TMY data
data, months, inputs, meta = pvlib.iotools.get_pvgis_tmy(
    latitude=1.3521, longitude=103.8198,
    map_variables=True,
)
data.to_csv('data/raw/pvgis_singapore_raw.csv')
```

Repeat for Oslo (59.9139, 10.7522) and Madrid (40.4168, -3.7038).

## Preprocessing pipeline

After downloading raw data, run the unified preprocessing pipeline:

```bash
python scripts/preprocess.py --site dkasc_site7  # or other site keys
```

This applies the eight-step pipeline from the paper:

1. **Period filter**: keep 2020–2023 (DKASC) or full TMY (PVGIS)
2. **Physical validity**: clip negative power to 0, cap humidity at 100%, drop negative irradiance
3. **Interpolation**: linear time-based interpolation for gaps ≤ 12 samples (1 hour at 5-min resolution)
4. **Aggregation**: resample to hourly mean
5. **Nighttime filter**: drop hours with GHI ≤ 10 W/m² (leaves ~17,261 daytime hours per DKASC site)
6. **z-score normalization**: per-channel, using **training split only**
7. **Sliding windows**: L=24, stride 1
8. **Chronological split**: 70/15/15 train/val/test with no temporal overlap

Output: `data/processed/<site>.csv` — single-file processed dataset ready for
`PVDatasetV2`.

## Expected file layout

```
data/
├── raw/                       # raw downloads (not committed; in .gitignore)
│   ├── dkasc_site7_raw.csv
│   ├── dkasc_site10_raw.csv
│   ├── dkasc_site2_raw.csv
│   ├── pvgis_singapore_raw.csv
│   ├── pvgis_oslo_raw.csv
│   └── pvgis_madrid_raw.csv
└── processed/                 # processed (not committed; in .gitignore)
    ├── dkasc_site7.csv
    ├── dkasc_site10.csv
    ├── dkasc_site2.csv
    ├── pvgis_singapore.csv
    ├── pvgis_oslo.csv
    └── pvgis_madrid.csv
```

## Verification

After preprocessing, verify the window counts:

| Split | n windows (DKASC) |
|-------|-------------------|
| Train | 12,109 |
| Val   | 2,575 |
| Test  | 2,577 |
| **Total** | **17,261** |

These should sum to the number of daytime hours per site, confirming the
stride-1 windowing recovers the full daytime-hour count.
