"""
Wrapper for Hugging Face Spaces deployment
Imports the server from python_dash_IA.py
"""
import sys
import importlib.util

# Load python_dash_IA.py as a module
spec = importlib.util.spec_from_file_location("python_dash_IA", "python_dash_IA.py")
python_dash_IA = importlib.util.module_from_spec(spec)
sys.modules["python_dash_IA"] = python_dash_IA
spec.loader.exec_module(python_dash_IA)

# Export the server for gunicorn
server = python_dash_IA.server
