# Contributing to ClearFrame

Thanks for helping. ClearFrame is free software for everyone, so contributions must keep it
free: no paid services, accounts, telemetry, remote inference or dependencies under
non-open-source licenses (`cargo deny check licenses` enforces this for Rust crates).

## Ground rules

- Never overwrite or modify a user's source video. Never delete projects, presets, models or
  exports as a side effect.
- Never download anything the user did not ask for. Never move user data without
  confirmation.
- Never mark something as working because a button exists: features need tests that
  exercise real behaviour.
- Tests must use temporary folders (`tempfile`, `--data-root`, `mktemp -d`) and must never
  touch a real user's ClearFrame data.
- The UI must be honest. Show what was measured, what is estimated and what is not
  available on this machine.

## Development

```sh
cargo test --workspace --exclude clearframe-desktop
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cd apps/desktop && npm ci && npm run typecheck && npm test && npx playwright test e2e/app.spec.ts
```

`npx tauri dev` runs the app. `npx vite` alone runs the interface in a browser against a
demo backend (`src/lib/mock.ts`). This is useful for UI work; the app labels it as demo mode.

If you change something the UI receives from Rust (settings, operation defaults, catalog or
presets), regenerate the UI fixtures:

```sh
CF_UPDATE_FIXTURES=1 cargo test -p clearframe-app --test ui_fixtures
```

If you change dependencies, run `python3 tools/gen_notices.py`. If you change the catalog,
regenerate `MODEL_LICENSES.md` (a test checks it).

## Adding a model

1. Implement or reuse an architecture in `crates/inference/src/arch` and prove parity with
   the official implementation (`tools/parity`).
2. Add a catalog entry with HTTPS URLs from the original publisher, SHA-256, sizes, licenses,
   training-data notes and limitations. Only add models whose license permits free use and
   redistribution of results for personal use. State commercial restrictions accurately.
3. Run `clearframe catalog validate model-catalog/catalog.json` and the model tests with
   `CLEARFRAME_TEST_MODELS=1`.

## Pull requests

Keep changes focused. Describe what you tested and on which OS and hardware. Report
anything you could not test.
