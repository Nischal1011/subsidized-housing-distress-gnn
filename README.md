# A Graph Neural Network Approach to Physical Distress Screening in Subsidized Housing

Code accompanying the paper **"A Graph Neural Network Approach to Physical Distress Screening in Subsidized Housing"**, by Nischal Subedi (University of Delaware) and Danush K. Wijekularathna (Troy University), accepted at *Data & Policy* (Cambridge University Press).

---

## Contents

* [Overview](#overview)
* [Key Ideas](#key-ideas)
* [Results at a Glance](#results-at-a-glance)
* [Relationship to the Published Results](#relationship-to-the-published-results)
* [Repository Structure](#repository-structure)
* [Setup](#setup)
* [Reproducing the Analysis](#reproducing-the-analysis)
* [Data & Model Availability](#data--model-availability)
* [License](#license)
* [Contact](#contact)

## Overview

Housing agencies must decide which of tens of thousands of subsidized properties to prioritize for physical inspection, while systematic condition data exists for only a fraction of the inventory. This project develops a screening framework using **Graph Attention Networks (GATs)** to propagate physical distress signals from inspected to uninspected properties through ownership, spatial, and census-tract networks, then integrates those predictions with an economic pressure measure (the *rent gap* between market and restricted rents) to identify properties facing compounded preservation threats.

## Key Ideas

* **Graph-based screening for physical distress.** A GAT trained on the subset of properties with HUD physical inspection scores estimates distress risk for the much larger set of properties that have never been inspected, leveraging the network structure created by shared ownership, geographic proximity, and shared census tracts.
* **Portfolio-level landlord behavior.** The model is evaluated for whether ownership connections carry predictive signal beyond what neighborhood characteristics alone would suggest, testing a "problem landlord" hypothesis from the housing policy literature.
* **Cross-program generalization.** Most subsidized housing (state-administered programs like LIHTC) never receives federal physical inspection. The paper tests how well a model trained on federally inspected properties (Section 8) transfers to LIHTC properties, and finds this generalization is severely limited, a finding with direct implications for deployment.
* **Compounded risk screening.** Physical distress predictions are combined with rent gap, subsidy expiration timelines, and property age into a multi-dimensional framework for identifying properties facing several simultaneous preservation threats at once, and characterizing their geographic concentration.

## Results at a Glance

* The GAT-based screening approach improves prioritization efficiency over conventional (non-graph) baselines and over simple heuristics like property age, concentrating physical distress risk among top-ranked properties.
* Ownership-network structure carries information about distress risk beyond what neighborhood/tract characteristics alone capture, consistent with portfolio-level clustering of landlord behavior.
* Cross-program transfer (training on federally inspected properties, evaluating on state-administered LIHTC properties) shows a substantial performance drop, underscoring that program-specific model development is needed before operational deployment outside HUD-administered programs.
* Crossing physical-distress predictions with economic pressure, subsidy-expiration timelines, and property age identifies a small set of properties facing multiple simultaneous preservation threats, which cluster geographically in a handful of states.

Exact figures are reported in the paper.

## Relationship to the Published Results

The repository contains the code and aggregated outputs associated with the analyses reported in this article. Because neural-network training is stochastic, retraining may produce numerical variation across software, hardware, and dependency versions. Repository outputs are expected to directionally align with the manuscript's reported patterns and conclusions; exact values may differ slightly for this reason. Where outputs diverge from the manuscript, the peer-reviewed article remains the authoritative source.

## Repository Structure

```text
.
├── src/
│   ├── pipeline/       # Core, order-dependent pipeline: data prep -> baseline model -> GNN variants
│   ├── analysis/       # Downstream tables, figures, and robustness analyses built on pipeline outputs
│   └── utils/          # Standalone utilities (e.g. Census/ACS data download)
├── outputs/            # Figures and summary tables backing the paper
├── requirements.txt
└── README.md
```

Large per-property score files and trained model weights are not included. See [Data & Model Availability](#data--model-availability).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch, and PyTorch Geometric.

## Reproducing the Analysis

Run the scripts in `src/pipeline/` in order from the repository root:

| Step | Script                          | Purpose                                                                      |
| ---- | -------------------------------- | ---------------------------------------------------------------------------- |
| 1    | `01_data_preparation.py`         | Merges NHPD property records with ACS tract-level characteristics            |
| 2    | `02_baseline_xgboost_model.py`   | Trains the XGBoost baseline model with SHAP explanations                     |
| 3    | `03_gnn_semi_supervised.py`      | Earlier graph-model iteration                                                |
| 4    | `04_gnn_enhanced.py`             | Earlier graph-model iteration                                                |
| 5    | `05_gnn_final_model.py`          | The final Graph Attention Network used for the paper's headline results      |
| 6    | `06_validate_hidden_risk.py`     | Sanity-checks the distress-rate gap between labeled and unlabeled properties |

`src/analysis/` contains the scripts that turn pipeline outputs into the paper's specific tables and figures. Most notably:

* `baseline_comparison.py` — trains and evaluates the GAT alongside Logistic Regression and XGBoost on an identical train/validation/test split, for a direct model comparison
* `cross_program_validation.py` — the Section 8 → LIHTC generalization test
* `analyze_urban_rural_perf.py` — performance stratified by urban/suburban/rural context
* `integration_table.py` / `paper_analysis.py` — the compounded-risk and screening-category analyses
* `table4_edge_ablation.py` — ablation study isolating the contribution of each edge type
* `table8_attention_weights.py` — learned attention weight by edge type

See each script's docstring for full details on what it produces.

## Data & Model Availability

This repository contains code and aggregated outputs, not the underlying source data or trained model weights, in line with the paper's data availability statement:

* **REAC physical inspection scores** are publicly available from HUD Multifamily Housing and HUD User.
* **National Housing Preservation Database (NHPD)** property-level data are available through the NHPD website, subject to registration.
* **Census tract characteristics** are from the American Community Survey 2017–2021 5-year estimates and are publicly available.
* **Property-level details** for the highest-risk properties identified in the paper are available from the corresponding author upon reasonable request.

To reproduce the analysis, place the corresponding data files in a local `data/` directory. See each pipeline script's `CONFIG['input_file']` for expected filenames and run the pipeline in order.

## License

This code is released under the [MIT License](LICENSE).

## Contact

Questions about the paper or this repository: [nsubedi@udel.edu](mailto:nsubedi@udel.edu)
