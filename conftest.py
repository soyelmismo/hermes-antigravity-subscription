"""Keep the suite hermetic against two host facts that leak into the registry.

1. ``$HERMES_HOME/plugins/`` can hold a second copy of this plugin. Hermes runs
   provider discovery on the first ``get_provider_profile()`` call and
   re-registers that copy under the same name, so the registry answers with the
   installed copy instead of the source tree under test. Every field difference
   between the two copies then fails assertions that pass on a clean checkout.

2. The repository root is itself a Python package (the plugin entry point is
   ``__init__.py``), so pytest imports it while collecting, and the test modules
   import it again under their own names. Each import runs
   ``register_provider()`` and the registry is last-writer-wins, so which copy
   answers depends on collection order.

``HERMES_HOME`` is pointed at an empty throwaway directory for (1). For (2) the
plugin is loaded once, under the canonical module name
``antigravity_plugin_entry``, and an autouse fixture re-registers that profile
before every test so the source tree under test always wins.

This file is imported by pytest before any test module, so the isolation is in
place before the first ``import providers``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent
CANONICAL_MODULE = "antigravity_plugin_entry"
PROFILE_NAME = "antigravity-subscription-directsdk"

_ISOLATED_HOME: Path | None = None


def _isolate_hermes_home() -> Path:
    """Point HERMES_HOME at an empty directory so discovery finds no plugins."""
    global _ISOLATED_HOME
    if _ISOLATED_HOME is None:
        _ISOLATED_HOME = Path(tempfile.mkdtemp(prefix="hermes-antigravity-tests-"))
        os.environ["HERMES_HOME"] = str(_ISOLATED_HOME)
    return _ISOLATED_HOME


def load_plugin_package():
    """Import the plugin entry point once, under CANONICAL_MODULE."""
    module = sys.modules.get(CANONICAL_MODULE)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        CANONICAL_MODULE,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover - layout guard
        raise ImportError(f"cannot load {PLUGIN_DIR / '__init__.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[CANONICAL_MODULE] = module
    spec.loader.exec_module(module)
    return module


_isolate_hermes_home()
PLUGIN = load_plugin_package()


@pytest.fixture(autouse=True)
def _source_tree_profile_wins():
    """Re-register the source tree's profile so a stray import cannot win."""
    import providers

    profile = PLUGIN.antigravity_profile
    if providers.get_provider_profile(PROFILE_NAME) is not profile:
        providers.register_provider(profile)
    yield


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001 - pytest hook signature
    """Drop the throwaway home once the suite is done."""
    global _ISOLATED_HOME
    if _ISOLATED_HOME is not None:
        shutil.rmtree(_ISOLATED_HOME, ignore_errors=True)
        _ISOLATED_HOME = None
