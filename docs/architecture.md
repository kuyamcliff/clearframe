# Architecture

```
React UI (apps/desktop/src) ──IPC──▶ Tauri commands (apps/desktop/src-tauri)
                                          │
clearframe CLI (crates/cli) ──────────────┤
                                          ▼
                          clearframe-app: App services
          queue/scheduler · model installs · storage · resources · previews
                                          │ spawns
                                          ▼
                 worker process (same executable, --worker / --still)
                          clearframe-pipeline runner
               ┌──────────────────┼──────────────────────┐
         clearframe-media    clearframe-inference    clearframe-models
         FFmpeg processes    candle model code       safe weight loading
```

## Processes

The desktop app and the CLI are thin front ends over `clearframe_app::App`. Every export and
every still preview runs in a **separate worker process**, which is the same executable
started with `--worker <job.json>` or `--still <request.json> <dir>`. Workers report progress
as JSON lines on stdout and take pause/cancel commands on stdin. A crash, a GPU driver fault
or running out of memory in a worker cannot take down the UI. When a worker dies, the job
becomes *recoverable* and resumes later from its last finished segment.

## Jobs

1. **Plan** (`pipeline::plan`): the planner propagates the frame format (size, rate,
   interlacing, transfer) through the stages. It checks parameters, model availability,
   FFmpeg capabilities and memory, and splits the work into passes. Stabilization needs a
   separate analysis pass when it follows model stages. Blocking problems are reported
   before anything starts.
2. **Run** (`pipeline::runner`): FFmpeg decodes to RGB48 over a pipe. Native stages run in
   Rust (models, retiming, grain) and FFmpeg filter stages run in filter processes. A second
   FFmpeg encodes into Matroska segments of about 10 s each.
3. **Assemble**: segments are joined with the concat demuxer using exact packet durations.
   Audio, subtitles, chapters and metadata are muxed from the source (stretched or muted for
   slow motion).
4. **Validate**: the output is probed and fully decoded.
5. **Finalize**: the file is hard-linked or renamed into place without ever replacing an
   existing file.

Pausing stops at a segment boundary. Resuming counts the packets already written and seeks
the decoder with pre-roll, so a paused-and-resumed export is bit-identical to an
uninterrupted one (tested).

## Models

`model-catalog/catalog.json` lists every model with HTTPS URLs, SHA-256 hashes, sizes,
licenses and provenance. Catalog updates must carry an Ed25519 signature from a key in
`trusted_keys.json`. Installing a model takes these steps:

1. Disk space is checked.
2. The download is resumable and respects proxies and NO_PROXY.
3. Hashes are verified.
4. Archive members are extracted by name only (nothing is executed, paths cannot escape).
5. Pickle checkpoints are read with a restricted unpickler. It only accepts tensor
   reconstruction from an allow-list and never runs code.
6. Weights are converted to safetensors.
7. A smoke test runs the model on a synthetic image.
8. The finished folder is swapped in atomically.

Only after all of that does the model count as installed. The filesystem is the source of
truth: missing or damaged folders are reported as such, never silently re-downloaded.

## Inference

The `clearframe-inference` crate implements SRVGGNetCompact, RRDBNet, FFDNet, DRUNet,
NAFNet, RIFE 4.26, RetinaFace and GFPGAN 1.4 with candle. Parity with the official PyTorch
code is checked by `tools/parity` (max abs error ≤ 3e-5). Large frames are processed in
overlapping tiles sized from the memory budget. An out-of-memory error on a device retries
with smaller tiles.

## Storage and resources

See [storage.md](storage.md). Resource limits (threads, priority, memory budget, device) are
resolved from Preferences → Performance into a `ResourcePlan` and passed to each worker. The
worker configures rayon, FFmpeg `-threads`/`-filter_threads`, process priority and tile
sizes from it. The scheduler starts jobs only while their estimated memory fits the budget.
It pauses them at a safe point under system memory pressure, or when free disk space drops
below the safety margin.
