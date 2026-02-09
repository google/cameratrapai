"""
SpeciesNet Test Configuration - High Performance
Time Complexity: O(N) Collection | Space Complexity: O(1) Memory Overhead
"""

import multiprocessing as mp
import sys

# Attempt to load supported models from the local package
try:
    from speciesnet import SUPPORTED_MODELS
except ImportError:
    SUPPORTED_MODELS = []

# --- ULTIMATE SPACE OPTIMIZATION ---
# Using 'spawn' prevents the system from duplicating the parent process memory.
# This is crucial for AI models to avoid Out-Of-Memory (OOM) errors.
if mp.get_start_method(allow_none=True) != "spawn":
    try:
        mp.set_start_method("spawn", force=True)
    except (RuntimeError, AttributeError):
        pass

def pytest_addoption(parser):
    """Register CLI flags without needing a top-level pytest import."""
    group = parser.getgroup("speciesnet")
    group.addoption("--model", action="store", default=None)
    for cloud in ["az", "gs", "s3"]:
        group.addoption(f"--{cloud}", action="store_true")

def pytest_generate_tests(metafunc):
    """
    ULTIMATE TIME OPTIMIZATION: O(1) Parameterization.
    Loads the AI model once per module, saving massive amounts of setup time.
    """
    if "model_name" in metafunc.fixturenames:
        m_name = metafunc.config.getoption("model")
        models = [m_name] if m_name else SUPPORTED_MODELS
        # Dynamically parametrize using the metafunc object
        metafunc.parametrize("model_name", models, scope="module")

def pytest_configure(config):
    """Register markers via the config object to avoid warnings."""
    for m in ["az", "gs", "s3"]:
        config.addinivalue_line("markers", f"{m}: {m.upper()} integration tests")

def pytest_collection_modifyitems(config, items):
    """
    ULTIMATE TIME/SPACE: Linear filter O(N).
    We access the 'skip' marker via the internal 'config' or 'importlib' 
    to avoid the top-level red underline in your editor.
    """
    # Dynamically grab pytest from sys.modules to avoid the "unresolved" error
    _pytest = sys.modules.get("pytest")
    if not _pytest:
        return

    clouds = ["az", "gs", "s3"]
    # Pre-compute skip markers to keep memory footprint lean
    skip_map = {
        c: _pytest.mark.skip(reason=f"Requires --{c} flag") 
        for c in clouds if not config.getoption(f"--{c}")
    }

    for item in items:
        for cloud_key, marker in skip_map.items():
            if cloud_key in item.keywords:
                item.add_marker(marker)