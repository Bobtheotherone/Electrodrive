from __future__ import annotations

import torch

from electrodrive.gfn import Action, ElectrodriveProgramEnv, Grammar
from electrodrive.gfn.env import PartialProgramState, SpecMetadata
from electrodrive.utils.device import get_default_device


def _make_env(max_length: int = 8, min_length_for_stop: int = 1) -> ElectrodriveProgramEnv:
    return ElectrodriveProgramEnv(grammar=Grammar(), max_length=max_length, min_length_for_stop=min_length_for_stop)


def _make_spec() -> SpecMetadata:
    return SpecMetadata(geom_type="plate", n_dielectrics=1, bc_type="dirichlet")


def test_encode_decode_round_trip_matches_action() -> None:
    env = _make_env()
    action = Action(
        action_type="add_primitive",
        action_subtype="baseline",
        discrete_args={"family_name": "baseline", "conductor_id": 2, "motif_id": 1},
    )
    encoded = env.encode_action(action)
    assert encoded.device.type == get_default_device().type
    decoded = env.decode_action(encoded)
    assert decoded.action_type == action.action_type
    assert decoded.discrete_args.get("family_name") == "baseline"
    assert decoded.discrete_args.get("conductor_id") == 2


def test_action_mask_determinism_and_stop_rules() -> None:
    env = _make_env()
    state = env.reset(spec="spec", spec_meta=_make_spec(), seed=123)
    mask1 = env.get_action_mask(state)
    mask2 = env.get_action_mask(state)
    assert torch.equal(mask1, mask2)
    assert mask1.device.type == get_default_device().type
    assert env.stop_index is not None
    assert bool(mask1[env.stop_index].item()) is False  # stop disallowed before min length


def test_mask_cuda_cache_reuse() -> None:
    env = _make_env()
    state = env.reset(spec="spec", spec_meta=_make_spec(), seed=5)
    mask1 = env.get_action_mask(state)
    mask2 = env.get_action_mask(state)

    assert mask1.device.type == env.device.type
    assert state.mask_cuda is not None
    assert state.mask_cache_key == env._mask_cache_key(state)
    assert mask1.data_ptr() == mask2.data_ptr()


def test_cache_invalidation_on_step() -> None:
    env = _make_env()
    state = env.reset(spec="spec", spec_meta=_make_spec(), seed=8)
    mask1 = env.get_action_mask(state)
    prev_ptr = mask1.data_ptr()

    action = next(a for a in env.actions if a.action_type == "add_primitive")
    next_state, _, _ = env.step(state, action)
    assert next_state.mask_cuda is None
    assert next_state.mask_cache_key is None

    mask2 = env.get_action_mask(next_state)
    assert mask2.data_ptr() != prev_ptr
    assert next_state.mask_cache_key == env._mask_cache_key(next_state)


def test_reverse_round_trip_restores_state_hash() -> None:
    env = _make_env(max_length=8)
    state = env.reset(spec="spec", spec_meta=_make_spec(), seed=0)
    actions = [
        Action(
            action_type="add_primitive",
            action_subtype="baseline",
            discrete_args={"family_name": "baseline"},
        ),
        Action(action_type="add_pole", discrete_args={"interface_id": 0, "n_poles": 1}),
        Action(action_type="stop"),
    ]
    states = [state]
    for action in actions:
        state, done, _ = env.step(state, action)
        states.append(state)
        if action.action_type != "stop":
            assert done is False
    assert env.is_terminal(state) is True
    assert states[-1].state_hash != states[0].state_hash

    for idx, action in enumerate(reversed(actions), start=1):
        state = env.reverse_step(state, action)
        expected = states[-(idx + 1)]
        assert state.state_hash == expected.state_hash
    assert state.state_hash == states[0].state_hash
    assert states[0].program.canonical_bytes == state.program.canonical_bytes


def test_batch_step_and_tokens_on_device() -> None:
    env = _make_env()
    base_state = env.reset(spec="spec", spec_meta=_make_spec(), seed=1)
    action = Action(
        action_type="add_primitive",
        action_subtype="baseline",
        discrete_args={"family_name": "baseline"},
    )
    next_states, done = env.step_batch([base_state, base_state], [action, action])
    assert len(next_states) == 2
    assert done.dtype == torch.bool
    assert done.device.type == get_default_device().type
    assert all(ns.ast_token_ids is not None for ns in next_states)
    assert all(ns.ast_token_ids.device.type == get_default_device().type for ns in next_states)

    state_tokens = base_state.to_token_sequence(max_len=env.max_length, grammar=env.grammar)
    env_tokens = env._tokenize_program(base_state.program)
    assert torch.equal(state_tokens, env_tokens)


class VariantGrammar(Grammar):
    def enumerate_actions(self) -> tuple[Action, ...]:
        base_actions = [action for action in super().enumerate_actions() if action.action_type not in ("conjugate_pair", "stop")]
        conjugates = [
            Action(action_type="conjugate_pair", discrete_args={"block_ref": 0}),
            Action(action_type="conjugate_pair", discrete_args={"block_ref": 2}),
        ]
        stop = [Action(action_type="stop")]
        return tuple(base_actions + conjugates + stop)


def test_discrete_args_affect_legality() -> None:
    env = ElectrodriveProgramEnv(grammar=VariantGrammar(), max_length=4, min_length_for_stop=1)
    state = env.reset(spec="spec", spec_meta=_make_spec(), seed=7)
    build_action = Action(
        action_type="add_primitive",
        action_subtype="baseline",
        discrete_args={"family_name": "baseline"},
    )
    state, _, _ = env.step(state, build_action)

    allowed = Action(action_type="conjugate_pair", discrete_args={"block_ref": 0})
    rejected = Action(action_type="conjugate_pair", discrete_args={"block_ref": 2})
    mask = env.get_action_mask(state)
    allowed_idx = env.action_to_index[env._action_key(allowed)]
    rejected_idx = env.action_to_index[env._action_key(rejected)]
    assert bool(mask[allowed_idx].item()) is True
    assert bool(mask[rejected_idx].item()) is False

    env.step(state, allowed)
    try:
        env.step(state, rejected)
    except ValueError as exc:
        assert "Illegal action under current state mask" in str(exc)
    else:
        raise AssertionError("Expected illegal action to raise ValueError")


def test_mask_cpu_path_avoids_cuda_tensor_in_make_state() -> None:
    env = _make_env(max_length=6, min_length_for_stop=1)
    spec_meta = _make_spec()

    import electrodrive.gfn.env.program_env as program_env_module

    original_tensor = program_env_module.torch.tensor
    original_compute = env._compute_mask_cpu
    guard_state = {"active": False}

    def guarded_tensor(*args, **kwargs):  # type: ignore[no-untyped-def]
        device = kwargs.get("device")
        if guard_state["active"] and isinstance(device, torch.device) and device.type == "cuda":
            raise AssertionError("Unexpected CUDA tensor creation in program_env mask path")
        return original_tensor(*args, **kwargs)

    def wrapped_compute(self, state, base_mask):  # type: ignore[no-untyped-def]
        guard_state["active"] = True
        try:
            return original_compute(state, base_mask)
        finally:
            guard_state["active"] = False

    env._compute_mask_cpu = wrapped_compute.__get__(env, type(env))
    program_env_module.torch.tensor = guarded_tensor
    try:
        state = env.reset(spec="spec", spec_meta=spec_meta, seed=3)
        action = next(a for a in env.actions if a.action_type == "add_primitive")
        for _ in range(3):
            state, _, _ = env.step(state, action)
            assert isinstance(state.action_mask, tuple)
    finally:
        program_env_module.torch.tensor = original_tensor
        env._compute_mask_cpu = original_compute

    mask = env.get_action_mask(state)
    if torch.cuda.is_available() and env.device.type == "cuda":
        assert mask.device.type == "cuda"
    else:
        assert mask.device.type == "cpu"
