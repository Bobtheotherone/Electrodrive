import yaml


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_stage9_max_steps_aligned_with_training():
    train_cfg = _load_yaml("configs/stage9/train_gfn_rich_gate_proxy.yaml")
    pilot_cfg = _load_yaml("configs/stage9/discovery_stage9_pilot.yaml")
    push_cfg = _load_yaml("configs/stage9/discovery_stage9_push.yaml")

    max_length = int(train_cfg.get("max_length", 0))
    pilot_steps = int(pilot_cfg.get("model", {}).get("structure_policy", {}).get("max_steps", 0))
    push_steps = int(push_cfg.get("model", {}).get("structure_policy", {}).get("max_steps", 0))

    assert pilot_steps <= max_length + 2
    assert push_steps <= max_length + 2
