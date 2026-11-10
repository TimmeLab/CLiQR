"""The activity log lives in the context-free session store.

Background threads (hardware receiver, HDF5 writer, camera) log while the
browser tab that started the session may have been refreshed away. Those
lines -- often ERRORs -- must be visible in the new tab, and concurrent
writers must never lose lines.
"""
import threading

from utils import state


def test_log_written_from_thread_survives_refresh():
    t = threading.Thread(target=state.add_log_message, args=("ERROR: from thread",))
    t.start()
    t.join()
    state.log_messages.set([])  # a fresh kernel context starts empty
    state.rehydrate_reactives_from_session()
    assert any("ERROR: from thread" in m for m in state.log_messages.value)


def test_concurrent_log_writes_lose_nothing():
    state.clear_log()

    def writer(i):
        for j in range(200):
            state.add_log_message(f"t{i} m{j}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(state.session["log_messages"]) == 1600


def test_log_is_capped():
    state.clear_log()
    for i in range(state.LOG_MAX + 10):
        state.add_log_message(f"m{i}")
    log = state.session["log_messages"]
    assert len(log) == state.LOG_MAX
    assert log[-1].endswith(f"m{state.LOG_MAX + 9}")


def test_clear_log_clears_both():
    state.add_log_message("x")
    state.clear_log()
    assert state.session["log_messages"] == []
    assert state.log_messages.value == []
