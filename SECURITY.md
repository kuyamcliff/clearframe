# Security policy

## Reporting

Please report vulnerabilities privately through GitHub's “Report a vulnerability” (Security
tab) on https://github.com/kuyamcliff/clearframe. Do not open a public issue for security
problems. We aim to acknowledge reports within 7 days.

## Design

- **Model files are untrusted input.** PyTorch checkpoints are parsed by a restricted
  unpickler that only reconstructs tensors from an allow-list and never executes code.
  Weights are converted to safetensors before use. Archives are read member by member,
  never extracted wholesale and never executed. Member paths cannot escape the target folder.
- **Downloads are pinned.** Every catalog file has an HTTPS URL and a SHA-256 hash. A model
  is installed only after hash verification and a smoke test. Catalog updates must be signed
  with Ed25519 by a key in `model-catalog/trusted_keys.json`. That list is currently empty,
  so online catalog updates are refused until maintainers publish a key.
- **Media files are untrusted input.** FFmpeg runs as a separate process with argument
  vectors (no shell). Processing runs in worker processes separate from the UI.
- **No network by default.** The app contacts the network only to install models the user
  selects, or when the user checks for updates.
- **Desktop shell.** The web view has a strict CSP, no remote content, and asset access is
  limited to the cache folder and files the user opened.
- **Installer.** It installs per machine with least privilege at runtime: ClearFrame itself
  never runs elevated, and user data is created by the user's own process.

## Supported versions

Security fixes go into the latest release.
