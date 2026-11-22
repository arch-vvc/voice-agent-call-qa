# Call QA demo notes

- Run demo: `python -m callqa`
- Free check: `python -m callqa --check --backend offline`
- Callers: story set 20 (`--set story`), full set 31 (`--set full`). Sample data is synthetic, offline numbers are simulated.

Commands (run from `demos/call-qa`; add `--backend offline` to `eval` and `compare` unless you mean to spend claude-cli or API quota; `diff` and `calibrate` take no backend flag and use no quota):
- `python -m callqa eval --agent v2 --set full --repeats 3 --backend offline --no-save` (flags `--set`, `--repeats`, `--rules PATH|none`, `--created-at ISO`, `--threshold`)
- `python -m callqa eval --agent v1 --set full --repeats 3 --backend offline --created-at 2026-11-05T00:00:00+05:30 --out samples/run_records/full_v1.json` (reproducible sample record, byte identical on a rerun)
- `python -m callqa diff samples/run_records/full_v1.json samples/run_records/full_v2.json --gate` (exit 1, names holder_says_bhai); v2 to v3 exits 0
- `python -m callqa compare --set full --backend offline --no-save` (offline: prompt versions, not models)
- `python -m callqa calibrate` (judge against human labels; `--include-synthetic` for the invented example)

Guards: full set pass^3 is v1 55% (17/31), v2 71% (22/31), v3 97% (30/31). Prices are dated 2026-11-06: verify before quoting.
RBI rules are not legal advice: three verified, two UNVERIFIED. Do not mention dashboard features that are not in `callqa/static`.
Regenerate samples and goldens only with the commands in `README.md`; `tests/test_samples.py` checks them.
