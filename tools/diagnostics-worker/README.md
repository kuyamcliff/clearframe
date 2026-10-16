# Diagnostics relay

A Cloudflare Worker that receives the logs ClipFX Enhance sends (see `crates/app/src/telemetry.rs`)
and emails each one to you through Resend, from `logs@mail.clipfx.me`, with a summary (version,
OS, hardware, errors, crash) in the body and the full redacted log attached.

The Resend key stays in the Worker, never in the app (an app can be unpacked; a key inside it
would let anyone send mail as mail.clipfx.me), and the recipient is fixed here.

## Deploy (once)

```sh
npm i -g wrangler
wrangler login
wrangler secret put RESEND_API_KEY   # sending-only key restricted to mail.clipfx.me
wrangler secret put REPORT_TO        # e.g. admin@clipfx.me
wrangler deploy                      # prints https://clipfx-diagnostics.<account>.workers.dev
```

Then in GitHub → Settings → Secrets and variables → Actions → **Variables**, add
`CLIPFX_DIAGNOSTICS_URL` = that address. Releases built after that send reports.

Resend's free plan sends 100 emails a day; the app sends at most one report per hour per
computer (and one at startup with anything left from the last session).
