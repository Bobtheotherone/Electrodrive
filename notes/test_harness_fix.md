Reason: pytest -k step10_scaffold was collecting temp_AI_upgrade/ tests that import electrodrive.gfdsl and failing during collection.
Change: added root pytest.ini with norecursedirs entries for staging/, temp_AI_upgrade/, and common build/venv cache folders.
Validation: pytest -k step10_scaffold -q (no collection errors; 2 passed, 284 deselected).
