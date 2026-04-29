# Temporal Network-Based Early Warning of Online Community Behaviour in Social Platforms

> **Signed Temporal Graph Signal Processing (ST-GSP) Framework**  
> American International University-Bangladesh  
> Papri Saha · Tasnuva Afrin · Shuvro Sankar Sen · Ahnaf Abdullah Zayad · Sudipta Kumar Das

---

## Overview

This repository contains the complete implementation of the ST-GSP framework — a novel approach to early warning detection of inter-community conflict in online social platforms using signed temporal graph signal processing.

The framework was applied to a 10-year Reddit corpus (2014–2024) of 1,870,468 clean within-corpus signed interaction edges across 230 subreddits, achieving **94% prediction accuracy** with **zero false negatives** on a 33-month test set.

---

## Key Findings

| Finding | Result |
|---|---|
| Network balance | λmin ≥ 0 across all 132 months — Reddit self-organises toward structural balance |
| Conflict eras detected | 2 (2015–2016 and 2023–2024) + 1 flash event (May 2021) |
| High-risk months flagged | 26 / 132 (19.7%) |
| Prediction accuracy | 94% overall, 0 false negatives |
| Structural propagation lag | 2 months between conflict event and spectral manifestation |
| Top bridge subreddit | r/market76 (48 top-3 appearances across 132 months) |
| Validated ground truth events | 4 (2015 Blackout, 2018 Gun ban, 2023 Adult quarantines, 2023 API protest) |

---

## Repository Structure

```
.
├── pipeline.ipynb                          # Main analysis notebook
├── data/
│   ├── subreddits24/                       # Raw .zst subreddit dump files (231 files)
│   │   └── *_edges_2014_2025.csv          # Extracted per-subreddit edge CSVs
│   ├── reddit_signed_edges_2014_2025.parquet   # Raw merged dataset (10.17M edges)
│   └── reddit_signed_edges_clean_2014_2025.parquet  # Clean within-corpus dataset (1.87M edges)
├── outputs/
│   ├── dataset_overview.png               # Figure 1: Dataset characterisation (4 panels)
│   ├── network_july2019_v2.png            # Figure 2: Network graph — July 2019
│   ├── tension_over_time.png              # Figure 3a: STI time series
│   ├── tension_heatmap.png                # Figure 3b: STI heatmap (year × month)
│   ├── gft_filter.png                     # Figure 4: GFT multiscale decomposition
│   ├── bridge_detection.png               # Figure 5: Bridge subreddit heatmap
│   └── confusion_matrix.png               # Figure 6: Predictive model results
└── docs/
    ├── Methodology_STGSP_Final.docx       # Section 4: Methodology
    └── Results_Discussion_STGSP_Final.docx # Sections 5–7: Results, Discussion, Conclusion
```

---

## Pipeline Overview

```
Raw Pushshift Data (.zst)
        ↓
1. Data Collection & Preprocessing
   - 230 subreddits selected by interaction density + metadata availability
   - VADER sentiment analysis → ±1 sign binarisation
   - Corpus filter: retain only within-corpus TARGET nodes
   - Output: 1,870,468 clean signed edges (Parquet)
        ↓
2. Signed Temporal Network Construction
   - Monthly aggregation → 132 snapshots
   - Symmetric signed adjacency matrix A ∈ ℝⁿˣⁿ per month
   - CSR sparse format, float32
        ↓
3. Objective 1 — Structural Tension Index (STI)
   - Signed Laplacian: Ls = D − A
   - STI = λmin(Ls) per month
   - Solver: eigvalsh (n<500) / eigsh ARPACK (n≥500)
        ↓
4. Objective 2 — Multiscale GFT Filter
   - Randomized SVD: k=30 modes, n_iter=3
   - Low-pass (modes 1–9) vs High-pass (modes 10–30)
   - Tension Ratio = E_high / ‖f‖²
   - Alert threshold: mean + 1σ = 0.244
        ↓
5. Objective 3 — Bridge Detection & Prediction
   - Spectral leakage ratio per node
   - Top-10 bridge nodes per month
   - Logistic Regression on 6 GFT-derived features
   - Temporal split 75/25, class_weight='balanced'
```

---

## Dataset

**Source:** [Pushshift Reddit Archive](https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4) via Academic Torrents

| Property | Value |
|---|---|
| Raw edges | 10,179,780 |
| Clean within-corpus edges | 1,870,468 |
| Source subreddits | 230 |
| Unique target subreddits | 227 |
| Temporal coverage | January 2014 – December 2024 |
| Monthly snapshots | 132 |
| Positive edges (+1) | 1,332,738 (71.3%) |
| Negative edges (−1) | 537,730 (28.7%) |
| Min edges per month | 3,372 (February 2018) |
| Mean edges per month | 14,170 |

### Why 18.4% of raw edges were retained

The raw regex pattern `r/([A-Za-z0-9_]+)` extracted 10.17M edges targeting 423,083 unique strings. Inspection revealed 84% of target nodes were non-subreddit strings (hex codes, numeric IDs, transaction references) from trading subreddit posts. A corpus filter retaining only edges where TARGET ∈ known SOURCE subreddits reduced the dataset to 1,870,468 genuine within-corpus interactions.

---

## Requirements

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn networkx tqdm pyarrow
```

| Library | Version | Purpose |
|---|---|---|
| pandas | ≥ 2.0 | Data processing |
| numpy | ≥ 1.24 | Array computation |
| scipy | ≥ 1.10 | Sparse matrices, eigensolvers |
| scikit-learn | ≥ 1.3 | Randomized SVD, Logistic Regression |
| matplotlib | ≥ 3.7 | All visualisations |
| seaborn | ≥ 0.12 | Heatmaps |
| networkx | ≥ 3.0 | Network visualisation |
| tqdm | any | Progress bars |
| pyarrow | any | Parquet I/O |

**Hardware used:** Apple Mac Mini M4 (10-core CPU, unified memory)  
**Runtime:** ~4–6 minutes for full pipeline across 132 monthly snapshots

---

## Reproducing Results

### Step 1 — Data preparation
```python
import pandas as pd

# Load raw dataset
df = pd.read_parquet("data/reddit_signed_edges_2014_2025.parquet")

# Apply corpus filter
known_subreddits = set(df['SOURCE'].unique())
df_clean = df[df['TARGET'].isin(known_subreddits)].copy()
df_clean['datetime']   = pd.to_datetime(df_clean['TIMESTAMP'], unit='s')
df_clean['year_month'] = df_clean['datetime'].dt.to_period('M')

# Aggregate to monthly edges
monthly_edges = df_clean.groupby(
    ['year_month', 'SOURCE', 'TARGET', 'SENTIMENT']
).size().reset_index(name='interaction_count')
monthly_edges['sign'] = monthly_edges['SENTIMENT']
```

### Step 2 — Run full pipeline
Open and run `pipeline.ipynb` sequentially. All five stages are self-contained cells with progress bars.

### Step 3 — Expected outputs

**Objective 1 (STI):**
```
Active months: 32 / 132
Peak: July 2019, λmin = 0.143
Mean tension (all months): 0.019
```

**Objective 2 (GFT):**
```
Alert threshold: 0.244
High-risk months flagged: 26
Mean tension ratio: 0.169
```

**Objective 3 (Bridge + Prediction):**
```
Top bridge: market76 (48 appearances)
Model accuracy: 94%
False negatives: 0
```

---

## Validated Ground Truth Events

| # | Event | Date | Detection |
|---|---|---|---|
| 1 | Great Reddit Blackout | July 2015 | STI elevated Jan–Sep 2015 ✅ |
| 2 | Reddit Gun Accessory Ban | March 2018 | r/gunaccessoriesforsale sentiment → −1.0 ✅ |
| 3 | Adult Community Quarantines | January 2023 | r/dirtykikpals bridge collapse detected ✅ |
| 4 | Reddit API Pricing Protest | June 2023 | 2-month lag: Aug + Oct 2023 flagged ✅ |

**Key finding:** A consistent **2-month structural propagation lag** was observed between documented platform-wide conflict events and their spectral manifestation — providing an actionable early warning window for platform moderation.

---

## Limitations

- **Size-dilution effect:** Negative correlation (r = −0.601) between monthly graph size and tension ratio due to fixed-rank RSVD approximation. Future work should explore adaptive k selection.
- **Regex edge extraction:** Broad alphanumeric pattern required post-hoc corpus filtering. Direct API-based extraction would be more precise.
- **Propagation lag:** The 2-month lag was empirically observed but not formally modelled. Future work should estimate this parameter and incorporate it into the predictive model.
- **Non-SFW communities:** 0.73% of clean edges originate from non-SFW communities. Sensitivity analysis confirmed negligible impact on all spectral findings.

---

## Citation

If you use this code or dataset in your research, please cite:

```
Saha, P., Afrin, T., Sen, S. S., Zayad, A. A., & Das, S. K. (2025).
Temporal Network-Based Early Warning of Online Community Behaviour
in Social Platforms. American International University-Bangladesh.
```

---

## References

Key references for the ST-GSP framework:

- Heider, F. (1946). Attitudes and cognitive organization. *Journal of Psychology*, 21(1), 107–112.
- Cartwright, D., & Harary, F. (1956). Structural balance. *Psychological Review*, 63(5), 277–293.
- Shuman, D. I., et al. (2013). The emerging field of signal processing on graphs. *IEEE Signal Processing Magazine*, 30(3), 83–98.
- Ortega, A., et al. (2018). Graph signal processing. *Proceedings of the IEEE*, 106(5), 808–828.
- Halko, N., et al. (2011). Finding structure with randomness. *SIAM Review*, 53(2), 217–288.
- Hutto, C. J., & Gilbert, E. E. (2014). VADER. *ICWSM 2014*, AAAI Press.
- Watchful1. (2025). Pushshift Reddit Archive. Academic Torrents.

---

*Last updated: March 2026*
