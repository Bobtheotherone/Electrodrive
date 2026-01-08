from electrodrive.experiments.run_discovery import should_early_exit


def test_allow_not_ready_overrides_ramp_abort() -> None:
    assert should_early_exit(allow_not_ready=True, ramp_abort=True) is False
    assert should_early_exit(allow_not_ready=True, ramp_abort=False) is False
    assert should_early_exit(allow_not_ready=False, ramp_abort=True) is True
