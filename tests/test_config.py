from localagent.model.config import PARAM_BUDGET, ModelConfig


def test_yaml_configs_load_and_fit_budget():
    for name in ("tiny-30m", "small-90m"):
        cfg = ModelConfig.from_yaml(f"configs/model/{name}.yaml")
        cfg.assert_within_budget()  # raises if over 100M
        assert 0 < cfg.estimate_params() <= PARAM_BUDGET


def test_tiny_is_about_30m():
    cfg = ModelConfig.from_yaml("configs/model/tiny-30m.yaml")
    assert 20e6 < cfg.estimate_params() < 45e6


def test_head_divisibility_asserts():
    import pytest

    with pytest.raises(AssertionError):
        ModelConfig(d_model=384, n_heads=5)  # 384 not divisible by 5
