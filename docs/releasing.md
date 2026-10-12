# Releasing

Releases are built and published by `.github/workflows/release.yml`.

## One-time setup

Add the release signing key as a repository secret:

1. GitHub → the repository → **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `RELEASE_SIGNING_KEY`
3. Value: the secret key (one line of base64). Its public half is in
   `model-catalog/trusted_keys.json`, so the app trusts updates signed with it.

Keep the secret key only there (and, if you like, in a password manager). Anyone with it can
publish updates that installed copies will accept. If it leaks, generate a new pair with
`clipfx catalog keygen`, put the new public key in `trusted_keys.json`, ship a release, then
replace the secret.

## Cutting a release

1. Bump `version` in the root `Cargo.toml` (and `apps/desktop/src-tauri/tauri.conf.json`).
2. Commit, then tag with notes:
   ```sh
   git tag -a v1.1.0 -m "What's new in 1.1.0 (shown in the app's update bar and on GitHub)"
   git push origin v1.1.0
   ```
3. The workflow builds:
   - `ClipFX-Enhance-<v>-x64-setup.exe`: full Windows installer, with FFmpeg, for new users.
   - `ClipFX-Enhance-<v>-x64-update.exe`: app-only update (a few MB), installed over an existing copy.
   - `clipfx-enhance_<v>_amd64.deb`: Linux package.
   - `update.json` and `update.json.sig`: the signed manifest installed apps read from
     `releases/latest/download/update.json`.

   It also installs the full installer on a Windows runner, applies the update on top and runs the
   app's self-test, so a broken update never gets published.
