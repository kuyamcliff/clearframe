#!/usr/bin/env bash
# Runs every built-in preset end to end on a short generated clip and checks each output.
#
#   tools/e2e-presets.sh <clipfx-cli> <data-root> <device> [seconds]
#
# <device> is "cpu" or "gpu:N" (passed as CLIPFX_DEVICE, so a software GPU such as Microsoft
# WARP or Mesa lavapipe is used even though the app never picks one automatically).
# Models must already be installed in <data-root>. Needs ffmpeg and ffprobe on PATH.
set -euo pipefail
cli=$1
root=$2
device=$3
secs=${4:-0.5}
work="$root/e2e-$(echo "$device" | tr ':' '-')"
mkdir -p "$work"
clip="$work/clip.mp4"
ffmpeg -v error -y -f lavfi -i "testsrc2=size=192x108:rate=24" -f lavfi -i "sine=frequency=440" -t "$secs" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$clip"
# HDR to SDR needs an HDR source: 10-bit HEVC tagged BT.2020 / PQ.
hdr="$work/clip-hdr.mp4"
ffmpeg -v error -y -f lavfi -i "testsrc2=size=192x108:rate=24" -t "$secs" -c:v libx265 -pix_fmt yuv420p10le \
  -x265-params "log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc" \
  -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc "$hdr"

presets=(${E2E_PRESETS:-anime-4k anime-2x live-1080 restore-old smooth-60 slowmo-4x stabilize denoise deblur faces hdr-to-sdr film-grain})
failed=()
for p in "${presets[@]}"; do
  start=$(date +%s)
  src=$clip
  [ "$p" = hdr-to-sdr ] && src=$hdr
  if CLIPFX_DEVICE="$device" "$cli" --data-root "$root" run "$src" --preset "builtin.$p" -o "$work/out" --name "$p" >"$work/$p.log" 2>&1; then
    out=$(ls "$work/out/$p".* 2>/dev/null | head -1 || true)
    frames=$( [ -n "$out" ] && ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets,width,height -of csv=p=0 "$out" || echo "")
    if [ -n "$out" ] && [ -n "$frames" ]; then
      echo "ok    $p ($device) $(( $(date +%s) - start ))s -> $(basename "$out") [w,h,frames: $frames]"
      continue
    fi
    echo "FAIL  $p ($device): no valid output"
  else
    echo "FAIL  $p ($device): exit $?"
  fi
  tail -n 40 "$work/$p.log"
  failed+=("$p")
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "failed on $device: ${failed[*]}"
  exit 1
fi
echo "all ${#presets[@]} presets passed on $device"
