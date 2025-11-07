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

## Deploy to Hugging Face Spaces

### 1) Prerequisites
- A Hugging Face account (https://huggingface.co/join)
- Git installed on your machine

### 2) Create a Hugging Face Access Token
1. Go to **Settings > Access Tokens**
2. Click **New token**
3. Give it a name (e.g., "IA_Dash")
4. Select **write** permissions
5. Copy the token (you'll need it later)

### 3) Create the Space
1. Go to **Hugging Face > Spaces > New Space**
2. **Space name:** Choose a name (e.g., "interdependence-analysis")
3. **Space SDK:** Choose **Docker**
4. **Visibility:** Public or Private (your choice)
5. Click **Create Space**

### 4) Push your code to the Space

Using git (recommended):

```bash
# Navigate to your project directory
cd /path/to/IA_Dash

# Add Hugging Face remote (replace with your username and space name)
git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>

# Add and commit the necessary files
git add Dockerfile requirements.txt python_dash_IA.py table_hat_game.csv assets/ README.md
git commit -m "Deploy to Hugging Face Spaces"

# Push to Hugging Face (use your HF token as password when prompted)
git push hf main
```

**Alternative:** Drag and drop files directly in the Hugging Face Space web interface.

### 5) Wait for Build
Once pushed, Hugging Face will automatically build your Docker container. This may take 2-5 minutes. You can watch the logs in the Space's **Logs** tab.

### 6) Access Your App
Once built, your app will be available at:
```
https://huggingface.co/spaces/<your-username>/<your-space-name>
```

## Files Structure for Deployment

Required files:
- `Dockerfile` - Docker configuration
- `requirements.txt` - Python dependencies
- `python_dash_IA.py` - Main application
- `table_hat_game.csv` - Data file
- `assets/styles.css` - Styling

Optional:
- `README.md` - Documentation
- `.dockerignore` - Files to exclude from Docker build

## Troubleshooting

### App not starting?
Check the **Logs** tab in your Space for error messages.

### Port issues?
Ensure your Dockerfile exposes port 7860 (Hugging Face Spaces default).

### Missing files?
Make sure all necessary files (CSV, assets folder) are included in your git push or upload.

### CSS not loading?
Verify the `assets/` folder is included and the path in your code is correct.
