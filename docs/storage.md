# Storage locations and cache

ClearFrame writes only to the folders below and to the export folder you choose. It never
writes next to your source videos unless you choose that folder for exports, and it never
overwrites an existing file.

| What | Windows | macOS | Linux |
|---|---|---|---|
| Program | `C:\Program Files\ClearFrame` | `/Applications/ClearFrame.app` | `/usr/bin`, `/usr/lib` (package) or the AppImage |
| Settings, database, logs | `%LOCALAPPDATA%\ClearFrame` | `~/Library/Application Support/ClearFrame` | `~/.local/share/ClearFrame` (data), `~/.config/ClearFrame` (storage config) |
| Models (default) | `%LOCALAPPDATA%\ClearFrame\models` | `~/Library/Application Support/ClearFrame/models` | `~/.local/share/ClearFrame/models` |
| Cache (default) | `%LOCALAPPDATA%\ClearFrame\cache` | `~/Library/Caches/ClearFrame` | `~/.cache/ClearFrame` |
| Exports (default) | `Videos\ClearFrame Exports` | `Movies/ClearFrame Exports` | `~/Videos/ClearFrame Exports` (XDG videos folder) |

The models and cache folders can be moved in **Settings → Storage & cache**. The chosen
locations are stored in `storage.json` in the config folder. `--data-root <dir>` (CLI)
keeps everything under one folder.

## Models folder

```
models/
  enhancement/<model-id>/      weights/*.safetensors, manifest.json, metadata.json, checksums.json
  denoising/<model-id>/
  restoration/<model-id>/
  interpolation/<model-id>/
  face-recovery/<model-id>/
  other/<model-id>/
```

Each model is stored once, in the folder of its primary category. Installs are staged in
`cache/downloads` and moved into place only after verification and a smoke test. If a model
folder is deleted or edited outside ClearFrame, the Model Library shows it as *Files missing*
or *Damaged* and offers Reinstall/Repair. Nothing is downloaded automatically. Models are
never deleted to free space.

## Cache areas

| Area | Contents | Cleared by “Clear cache” |
|---|---|---|
| previews | still and clip preview renders | yes |
| thumbnails | timeline thumbnails and scrub frames | yes |
| proxies | browser-playable copies for comparison | yes |
| downloads | partial model downloads (resumable) | only if selected |
| temp | temporary files of child processes (TMP/TEMP point here) | yes |
| jobs | segments and intermediates of exports | finished jobs only, unless “also delete work files of paused or interrupted exports” is ticked |

Automatic cleanup applies the size limit and maximum age to regenerable areas. It never
touches work files of unfinished exports, and never touches projects, presets, models or
exports.

## Moving folders

A move:

1. Checks the destination (writable, not inside the source, enough space).
2. Copies every file while writing a journal.
3. Verifies sizes.
4. Switches the configuration.
5. Deletes the old copy.

Running jobs must be paused first. Previews and installs are blocked during the move. You
can cancel; the old folder stays in use until the switch. If ClearFrame is killed mid-move,
the journal is used on the next start to finish or roll back. The configuration only ever
points at a complete folder.

## Caches ClearFrame does not control

- GPU driver shader/kernel caches (for example NVIDIA's `ComputeCache`, the macOS Metal
  cache and Mesa's shader cache) live where the driver puts them.
- WebView2/WebKit keep their own small profile folders: `%LOCALAPPDATA%\io.github.kuyamcliff.clearframe`
  on Windows, `~/.local/share/io.github.kuyamcliff.clearframe` on Linux, and
  `~/Library/WebKit/io.github.kuyamcliff.clearframe` on macOS.
- FFmpeg writes nothing outside the folders ClearFrame passes to it.
