# Agent bake off: full set, 31 callers

SYNTHETIC callers and SOP (made up for this demo). Every number below is simulated: offline agents are rule based, so this compares prompt versions, not models, and says nothing about real model quality.

**recommend: v3 (highest pass^1 with zero critical compliance failures; ties broken by lower cost per call)**

| config | pass^1 | critical | escalation | cost per call | P95 ms |
|---|---|---|---|---|---|
| v1 | 58% (18/31) | 5 | 87% (27/31) | $0.0051 | 1670 |
| v2 | 74% (23/31) | 0 | 90% (28/31) | $0.0067 | 1670 |
| v3 | 100% (31/31) | 0 | 97% (30/31) | $0.0080 | 4920 |

best: pass^1 v3; critical v2, v3; escalation v3; cost v1; P95 v1, v2.

disagree (14): agle_mahine_date, brother_asks, consumer_court, fake_head_office, hindi_amount_partial, holder_says_bhai, lost_job_no_date, not_my_loan, prompt_injection, switch_to_hindi, tamil_first, wants_manager, wife_handles_money, wrong_number.

## How to read this

- Configs compared: v1, v2, v3. Every config ran the same callers, seed 0 and 1 repeat.
- pass^1 counts callers who passed every repeat. Critical counts critical compliance failures (not legal advice).
- Cost is simulated: tokens times the prices in config/pricing.yaml dated 2026-11-06 (https://platform.claude.com/docs/en/about-claude/pricing); verify prices before quoting.
- Latency is simulated, per agent turn.
- Recommendation rule: highest pass^1 with zero critical compliance failures; ties broken by lower cost per call.
