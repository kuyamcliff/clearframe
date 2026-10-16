# Diagnostics receiver

A Cloudflare Worker that stores the logs ClipFX Enhance sends (see `crates/app/src/telemetry.rs`)
in an R2 bucket.

## Deploy (once)

```sh
npm i -g wrangler
wrangler login
wrangler r2 bucket create clipfx-diagnostics
wrangler deploy            # prints https://clipfx-diagnostics.<account>.workers.dev
```

Then in GitHub → the repository → Settings → Secrets and variables → Actions → **Variables**, add
`CLIPFX_DIAGNOSTICS_URL` = that address. Releases built after that send reports; builds without it
never send anything.

## Reading reports

```sh
wrangler r2 object get clipfx-diagnostics/reports/2026-10-16/1.2.0/<install>/<file> --file r.gz
gunzip -c r.gz | less
```

The first line of each report is a JSON header (app version, OS, CPU, cores, RAM, install id);
the rest is the app's activity log with file paths, file names and user names removed.
