# Human labels

This folder holds the human verdicts on judged calls. A person reads a transcript, says pass or fail, and the file
records the judge's verdict next to it, so the judge can later be checked against people.

`human_labels.jsonl` is written only by `POST /api/labels` on the dashboard server. It starts empty (the file does
not exist until the first label), and nothing in the repo commits a label. The `.jsonl` files in this folder are
ignored by git (see the repo `.gitignore`), so real labels never get committed by accident.

There is no labelling page yet: the route exists, the page is not built.

## Format

UTF 8, one JSON object per line, schema version 1:

```json
{"schema_version": 1, "kind": "label", "label_id": "hex8", "created_at": "ISO", "record": "20261105-170724_v2.json",
 "run_id": "str", "persona_id": "str", "repeat": 0, "transcript_sha256": "str",
 "human_pass": false, "human_failing_turn": 3, "human_reason": "never offered a callback", "labeller": "bob",
 "judge_pass": true, "judge_failing_turn": null, "judge_reason": "str", "judge_source": "llm",
 "on_simulated_run": false, "synthetic": false}
```

The client sends only `record`, `persona_id`, `repeat`, `human_pass`, `human_failing_turn`, `human_reason` and
`labeller`. The server copies every judge field from the run record, so a client cannot forge them.
`judge_source` is `"llm"` or `"rules"` (an offline run).

Added beyond the plan's C9 field list: `judge_failure_kind`, the failure kind the record stored for that call
(`judge`, `forbidden`, `missing`, `misheard`, `compliance` or null for a pass). `judge_source` is `"llm"` only on a
non offline run when the verdict is the model judge's (a pass, or kind `judge`); a deterministic rule failure is
`"rules"` even on a live run. A call that crashed is refused (400): there is no judge verdict to compare.

## Rules

- `human_pass` is true or false, nothing else.
- `human_failing_turn` is optional, must be null on a pass, and otherwise is the 0 based index of an agent turn in
  that call's transcript.
- `human_reason` is at most 500 characters and 900 bytes once saved (Devanagari takes 3 bytes a character, so about
  300 of them; a quote takes 2). `labeller` is 1 to 40 characters and 60 bytes, saved trimmed. The limits are chosen
  so a valid label always fits one line, and a refusal names the field to shorten.
- `judge_reason` is copied from the record but cut to 160 bytes, so a long judge note never blocks a label.
  Line separators (U+0085, U+2028, U+2029) are written escaped, so every line is one line to any reader.
- A line over 2 KB is refused (400). If the file does not end in a newline (a torn earlier write), one is written
  first. A file at 5 MB refuses further labels (409): archive it and start a new one.
- Appends take a thread lock and `flock` where the platform has it, so simultaneous labels never interleave.
