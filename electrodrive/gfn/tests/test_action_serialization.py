from __future__ import annotations

from electrodrive.gfn.dsl.action import Action


def test_action_round_trip_tokenization() -> None:
    action = Action(
        action_type="add_primitive",
        action_subtype="baseline",
        discrete_args={"family_name": "baseline", "conductor_id": 1},
        continuous_args={"scale": 0.5},
    )
    token = action.to_token()
    restored = Action.from_token(token)
    assert restored == action
    assert token == action.to_token()  # deterministic
