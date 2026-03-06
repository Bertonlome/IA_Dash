"""
Wrapper for Hugging Face Spaces deployment
Loads python_dash_generic.py and exports the WSGI server.
"""
import sys
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "python_dash_generic",
    pathlib.Path(__file__).parent / "python_dash_generic.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules["python_dash_generic"] = mod
spec.loader.exec_module(mod)

# Export the server for gunicorn
server = mod.server
