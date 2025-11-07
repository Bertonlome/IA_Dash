---
title: Interdependence Analysis
emoji: 🔗
colorFrom: blue
colorTo: green
sdk: docker
app_file: python_dash_IA.py
pinned: false
---

# Interdependence Analysis — Dash App (Hugging Face Space)

- Framework: Plotly Dash
- Hosting: Hugging Face Spaces (Docker)
- Usage: Upload your CSV to visualize/edit parameters in-session. No server-side storage.

## Run locally

```bash
pip install -r requirements.txt
python python_dash_IA.py
```

Open your browser at http://localhost:8050

Required files:
- `Dockerfile` - Docker configuration
- `requirements.txt` - Python dependencies
- `python_dash_IA.py` - Main application
- `table_hat_game.csv` - Data file
- `assets/styles.css` - Styling

Optional:
- `README.md` - Documentation
- `.dockerignore` - Files to exclude from Docker build