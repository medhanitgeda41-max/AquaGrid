"""Startup import smoke test: verifies app/app.py and every module in app/components/ import with zero errors."""
import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
COMPONENTS_DIR = APP_DIR / "components"


def test_app_and_every_component_module_imports_cleanly():
    """Importing app.app and every .py module in app/components/ must succeed without ImportError or AttributeError."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(APP_DIR) not in sys.path:
        sys.path.append(str(APP_DIR))

    import simulator

    assert callable(simulator.get_simulator)
    assert callable(simulator.simulator_step_lock)
    assert hasattr(simulator, "STEP_LOCK")
    sim = simulator.get_simulator()
    assert hasattr(sim, "step_lock")

    app_mod = importlib.import_module("app.app")
    assert callable(app_mod.main)

    component_files = sorted(
        f.stem for f in COMPONENTS_DIR.glob("*.py") if f.stem != "__init__"
    )
    assert len(component_files) >= 10, f"Expected component modules in {COMPONENTS_DIR}"

    for stem in component_files:
        mod = importlib.import_module(f"components.{stem}")
        assert mod is not None

    ai_panel = importlib.import_module("components.ai_prompt_panel")
    assert callable(ai_panel.render_ask_aquagrid_chat)
    assert callable(ai_panel.render_ai_prompt_panel)
