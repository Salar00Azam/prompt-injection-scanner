# Contributing

Thanks for taking a look. Contributions are welcome, especially new payloads.

## Adding a payload

Add an object to the right category in `payloads.json`:

```json
{
  "id": "DI-013",
  "name": "Short descriptive name",
  "severity": "high",
  "payload": "The injection text",
  "inject_as": "user_input",
  "expected_vulnerable": "What a vulnerable model would do",
  "expected_secure": "What a secure model should do",
  "tags": ["technique"]
}
```

- Keep `id` prefixes consistent with the category (`DI-`, `II-`, `SP-`, ...).
- Use a unique canary string (for example `INJECTION_SUCCESS_DI013`) so the
  detector can flag a hit.
- Avoid putting the canary in text the model is asked to translate, summarize
  or repeat. A faithful answer would then contain the canary without the
  injection succeeding (see DI-005 under Known limitations in the README).
- Set `severity` to one of `critical`, `high`, `medium`, `low`.
- For multi-turn attacks, set `"multi_turn": true` and provide a
  `payloads_sequence` list instead of `payload`.

Validate before opening a PR:

```bash
python -m json.tool payloads.json > /dev/null
python scanner.py --list-categories
```

## Code

Keep it dependency-light and readable. Run `ruff check scanner.py` if you have it.
Improvements to detection accuracy (fewer false positives and misses) are
especially welcome.

## Scope

This is a testing tool for authorized use. Please do not submit payloads whose
only purpose is to cause real-world harm rather than to test a defense.
