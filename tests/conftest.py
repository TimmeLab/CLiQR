"""Shared pytest fixtures for the CLiQR test suite.

Why this file exists
--------------------
`utils.state.session` is a single plain-module-global dict that is shared across
every test in the process (it is the authoritative, context-immune copy of the
session-critical reactives). Production code now writes many more keys into it
through `set_session` — hardware handles, camera run-state, disk/stall warnings.
Any test that exercises one of those code paths therefore MUTATES this global,
and without cleanup that mutation leaks into whichever test runs next. That makes
assertions about "the session's default values" order-dependent and flaky.

The autouse fixture below snapshots the session before each test and restores it
afterward, so every test starts from the same baseline regardless of what ran
before it. This isolates tests from each other without each one having to
remember to clean up by hand.
"""
import copy

import pytest

from utils import state


@pytest.fixture(autouse=True)
def restore_session_state():
    """Snapshot `state.session` before a test and restore it afterward.

    We take a deep copy so nested structures (e.g. the per-sensor `sensor_states`
    dict of dataclasses) are captured by value, not by reference. After the test
    runs, we put the saved values back and re-mirror them into the reactives via
    `rehydrate_reactives_from_session`, leaving both halves of the persistence
    mechanism (global dict + reactives) at the pre-test baseline.
    """
    saved_session = copy.deepcopy(state.session)

    # Run the test.
    yield

    # Restore the authoritative global in place (same dict object, so any code
    # holding a reference to `state.session` still sees the restored contents).
    state.session.clear()
    state.session.update(saved_session)

    # Re-point this process's reactives at the restored values so the reactive
    # mirror matches the global again.
    state.rehydrate_reactives_from_session()
