from scripts.normalize_appworld_trajectories import _actions


def test_short_trajectory_can_retain_completion_without_bootstrap_calls() -> None:
    trace = [
        {"app": "supervisor", "api": "show_profile", "arguments": {}},
        {"app": "spotify", "api": "search_artists", "arguments": {"query": "Lily"}},
        {"app": "supervisor", "api": "complete_task", "arguments": {"status": "success"}},
    ]
    assert [f"{item['app']}.{item['api']}" for item in _actions(trace, 4)] == [
        "spotify.search_artists"
    ]
    assert [f"{item['app']}.{item['api']}" for item in _actions(trace, 4, include_completion=True)] == [
        "spotify.search_artists",
        "supervisor.complete_task",
    ]
