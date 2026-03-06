[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

# Interdependence Analysis Dashboard
**An Open Interactive Tool for Applying Interdependence Analysis in Human–Autonomy Teaming**

---

## Overview

The **Interdependence Analysis (IA) Dashboard** is an open-source web application built with [Plotly Dash](https://dash.plotly.com/) that supports the practice of **Interdependence Analysis (IA)**, a core method in **Coactive Design** for engineering effective Human–Autonomy Teams (HAT).

The dashboard provides an interactive environment for loading, visualizing, and editing IA tables — structured matrices that map a task analysis to Observability, Predictability, and Directability (OPD) teaming requirements needed for each team configuration alternative.

The tool is designed to facilitate the jump from theory to practice: rather than working through IA tables in spreadsheets, users can directly interact with a purpose-built interface.

---

## Key Features

- **Parametric team configuration** — auto-detects team structure and agent roles from any CSV.
- **Interactive IA table editor** — edit coordination levels cell-by-cell directly in the browser
- **OPD requirements panel** — structured input fields for specifying observability, predictability, and directability needs per task
- **Capacity charts** — aggregated bar charts summarizing capacities to execute tasks by agent and by color level
- **Interdependence pie charts** — visual overview of the interdependence level across the joint activity
- **Automation proportion metric** — computed according to Liu & Kaber (2025)
- **CSV import / export** — load your own IA table; export the edited version at any time
- **Bundled example** — ships with a sample IA table (`V7/IA_V7.csv`) based on an smart-electronic-checklist in an aviation scenario

---

## Installation

### Option 1 – Run Locally

**Prerequisites:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/Bertonlome/Interdependence-Analysis.git
cd Interdependence-Analysis

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install dash pandas plotly

# Run the app
python python_dash_generic.py
```

Open your browser at [http://localhost:8050](http://localhost:8050)

---

### Option 2 – Deploy via Docker (Hugging Face Spaces or self-hosted)

A production-ready Docker configuration is provided under `hf_deployment/`.

```bash
cd hf_deployment
docker build -t ia-dashboard .
docker run -p 7860:7860 ia-dashboard
```

Refer to `hf_deployment/README.md` for full instructions on deploying to a Hugging Face Space.

---

## CSV Format

The dashboard auto-detects team structure from any CSV that follows this convention:

| Column | Description |
|---|---|
| `Row` | Row index (optional) |
| `Procedure` | Procedure or phase label |
| `Task Object` | Task or sub-task or required capacities description |
| `AgentName*` | Performer column — column name ending with `*` |
| `AgentName` | Supporter column — same base name, no `*` |
| `OPD Requirements` | Free-text field for coordination requirements |

Color values in agent columns must be one of: `green`, `yellow`, `orange`, `red`.

The bundled example (`V7/IA_V7.csv`) provides a reference for structuring your own IA table.

---

## Theoretical Background

This tool is part of the **Coactive Design** process developed by Johnson, and aims to guide designer of human-autonomy team beyond the substition-based automation philosophy, by focusing on interactions that promotes mandatory and optional dependencies between team-members

The IA also lead the designer to ask, for every task and every team configuration alternative:

> *What needs to be observable from whom, what needs to be predictable from whom, and what needs to be directable to whom?*

The color coding scheme:

| Color | Meaning for Performer| Meaning for Supporter |
|---|---|---|
| 🟢 Green | I can do it all | My assistance could improve efficiency |
| 🟡 Yellow | I can do it all but my reliability is < 100% | My assistance could improve reliability |
| 🟠 Orange | I can contribute but need assistance | My assistance is required |
| 🔴 Red | I cannot do it | I cannot provide assistance |

---

## Current Development Status

The platform is under active development.

Planned improvements include:

- Adding sequence constraints and what-ifs permutation
- Integration with task network modeling tools
- Export to structured report format (PDF / LaTeX)

Feedback from researchers, and practitioners is welcome.

---

## Citation

If you use this software for research or teaching, please cite:

> Berton, B., & Doyon-Poulin, P. (2026). *Interdependence Analysis Dashboard (Version 1.0)* [Software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

---

## License

This project is released under the **GNU-GPL v3 License**.  
See the [`LICENSE`](LICENSE) file for details.

---

## Contributing

Contributions are welcome.

---

## Contact

Benjamin Berton - benjaminberton64@gmail.com 
PhD Candidate – Cognitive Engineering  
Polytechnique Montréal
