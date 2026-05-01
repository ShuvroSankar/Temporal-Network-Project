# Temporal Network-Based Early Warning of Online Community Behaviour on Reddit

> **Signed Temporal Graph Signal Processing (ST-GSP) Framework**  
> American International University-Bangladesh  
> Shuvro Sankar Sen · Papri Saha · Sudipta Kumar Das · Tasnuva Afrin · Ahnaf Abdullah Zayad · Rajarshi Roy Chowdhury

---

## Overview

This repository contains the complete implementation of the ST-GSP framework, a novel approach to early warning detection of inter-community conflict in online social platforms using signed temporal graph signal processing.

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
├── pipeline.ipynb                               # Main analysis notebook (GFT pipeline)
├── data/
│   ├── subreddits24/                            # Raw .zst subreddit dump files (231 files)
│   │   └── *_edges_2014_2025.csv               # Extracted per-subreddit edge CSVs
│   └── reddit_signed_edges_2014_2025.parquet   # Clean within-corpus dataset (1.87M edges)
├── files/
│   ├── reddit.ipynb                             # .zst dump extraction → per-subreddit CSV files
│   └── marge_all_files.ipynb                   # Merges CSVs → clean Parquet file
├── outputs/
│   ├── dataset_overview.png                    # Dataset characterisation (4 panels)
│   ├── network_july2019_v2.png                 # Network graph — July 2019 (peak tension month)
│   ├── sti_over_time.png                       # STI time series (2 panels)
│   ├── network_tension_heatmap.png             # STI heatmap (year × month)
│   ├── high-risk_months.png                    # GFT multiscale decomposition
│   ├── Bridge_Subreddit_Leakage_Ratio_Over_Time # Bridge subreddit analysis (3 panels)
│   └── confusion_matrix.png                    # Predictive model results
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
   - Solver: eigvalsh (n < 500) / eigsh ARPACK (n ≥ 500)
        ↓
4. Objective 2 — Multiscale GFT Filter
   - Randomized SVD: k=30 modes, n_iter=3
   - Low-pass (modes 1–9) vs High-pass (modes 10–30)
   - Tension Ratio = E_high / ‖f‖²
   - Alert threshold: mean + 1σ = 0.244
        ↓
5. Objective 3 — Bridge Subreddit Detection
   - Spectral leakage ratio per node per month
   - Top-10 bridge nodes identified per monthly snapshot
   - Precursor analysis: 1–2 month lookahead window
        ↓
6. Predictive Modelling
   - Logistic Regression on 6 GFT-derived temporal features
   - Temporal train/test split 75/25, class_weight='balanced'
   - Z-score normalisation fitted on training set only
```

---

## Dataset

**Source:** [Pushshift Reddit Archive](https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4) via Academic Torrents

| Property | Value |
|---|---|
| Raw edges | 10,179,780 |
| Clean within-corpus edges | 1,870,468 (18.4% of raw) |
| Source subreddits | 230 |
| Unique target subreddits | 227 |
| Temporal coverage | January 2014 – December 2024 |
| Monthly snapshots | 132 |
| Positive edges (+1) | 1,332,738 (71.3%) |
| Negative edges (−1) | 537,730 (28.7%) |
| Min edges per month | 3,372 (February 2018) |
| Mean edges per month | 14,170 |

### Why 18.4% of raw edges were retained

The raw regex pattern `r/([A-Za-z0-9_]+)` extracted 10.17M edges targeting 423,083 unique strings. Inspection revealed 84% of target nodes were non-subreddit strings (hex codes, numeric IDs, transaction references embedded in trading community posts). A corpus filter retaining only edges where TARGET ∈ known SOURCE subreddits reduced the dataset to 1,870,468 genuine within-corpus interactions.

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
**Runtime:** ~4–6 minutes for full pipeline across 132 monthly snapshots (Joblib parallelisation)

---

## Reproducing Results

### Step 1 — Data preparation

Run `files/reddit.ipynb` to extract per-subreddit CSVs from the raw `.zst` Pushshift dump files, then run `files/marge_all_files.ipynb` to merge and filter them into the clean Parquet file.

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

Open and run `pipeline.ipynb` sequentially. All stages are self-contained cells with progress bars via `tqdm`.

### Step 3 — Expected outputs

**Objective 1 (STI):**
```
Active months (λmin > 0): 32 / 132
Peak: July 2019, λmin = 0.143
Mean tension (all months): 0.019
Mean tension (active months only): 0.082
```

**Objective 2 (GFT):**
```
Mean Tension Ratio: 0.169
Standard deviation: 0.075
Alert threshold (mean + 1σ): 0.244
High-risk months flagged: 26
```

**Objective 3 (Bridge Detection):**
```
Top bridge subreddit: r/market76 (48 top-3 appearances)
r/gunaccessoriesforsale: 30 appearances, 18 precursor instances
```

**Predictive Model:**
```
Overall accuracy: 94% (33-month test set)
High-risk recall: 1.00
High-risk precision: 0.88
False negatives: 0
```

---

## Validated Ground Truth Events

| # | Event | Date | Detection Method | Result |
|---|---|---|---|---|
| 1 | Great Reddit Blackout | July 2015 | Obj. 1 (STI) + Obj. 2 (TR) | STI elevated Jan–Sep 2015; 13 consecutive flagged months ✅ |
| 2 | Reddit Gun Accessory Sales Ban | March 2018 | Obj. 3 (bridge sentiment) | r/gunaccessoriesforsale mean sentiment → −1.0 (only subreddit to reach this value) ✅ |
| 3 | Adult Community Quarantines | January 2023 | Obj. 3 (bridge collapse) | r/dirtykikpals dropped from 143 edges (Dec 2022) to 11 edges (Jan 2023) ✅ |
| 4 | Reddit API Pricing Protest | June 2023 | Obj. 2 (2-month lag) | Aug 2023 TR=0.285, Oct 2023 TR=0.259 flagged ✅ |

**Key finding:** A consistent **2-month structural propagation lag** was observed between documented platform-wide conflict events and their spectral manifestation — providing an actionable early warning window for platform moderation that is unavailable to conventional edge-level sentiment monitoring.

---

## Limitations

- **Size-dilution effect:** Negative correlation (r = −0.601) between monthly graph size and Tension Ratio, due to fixed-rank RSVD approximation. Future work should explore adaptive k selection proportional to graph size.
- **Monthly temporal resolution:** Sub-monthly conflict dynamics are masked by the aggregation window. Daily or weekly snapshots during flagged periods may reveal finer-grained tension signatures.
- **Sample coverage:** The 230-subreddit corpus represents approximately 0.6% of Reddit's 40,000 most popular communities. Findings may not generalise to the full ecosystem.
- **Predictive model evaluation:** The logistic regression was evaluated on a single 33-month test set. Cross-validation on an independently collected dataset would strengthen confidence in the reported accuracy.
- **Non-SFW communities:** 19 non-SFW communities contribute 1.7% of clean edges. Sensitivity analysis confirmed negligible impact on all spectral findings (minimum monthly edges remain at 3,171 after exclusion).

---

## Citation

If you use this code or dataset in your research, please cite:

```
Sen, S. S., Saha, P., Das, S. K., Afrin, T., Zayad, A. A., & Chowdhury, R. R. (2025).
Temporal Network-Based Early Warning of Online Community Behaviour in Social Platform.
American International University-Bangladesh.
Preprint submitted to Elsevier.
```

---

## References

Key references for the ST-GSP framework:

- Heider, F. (1946). Attitudes and cognitive organization. *Journal of Psychology*, 21(1), 107–112.
- Cartwright, D., & Harary, F. (1956). Structural balance: A generalization of Heider's theory. *Psychological Review*, 63(5), 277–293.
- Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013). The emerging field of signal processing on graphs. *IEEE Signal Processing Magazine*, 30(3), 83–98.
- Ortega, A., Frossard, P., Kovačević, J., Moura, J. M. F., & Vandergheynst, P. (2018). Graph signal processing: Overview, challenges, and applications. *Proceedings of the IEEE*, 106(5), 808–828.
- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217–288.
- Hutto, C. J., & Gilbert, E. E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of the 8th ICWSM*. AAAI Press.
- Watchful1. (2025). Subreddit comments/submissions 2005–06 to 2024–12. Academic Torrents.

---

*Last updated: May 2026*
