#!/usr/bin/env bash
# Builds a corpus of real-footage test videos in the formats people actually bring:
# phone, camera, DVD, web, editing and HDR files, plus awkward cases (rotation, variable frame
# rate, odd sizes, no audio, several audio tracks, non-ASCII names).
#
#   tools/real-corpus.sh <out-dir> [live-action-source] [animation-source]
#
# Sources default to the Blender Foundation's open films (CC BY 3.0, downloaded if missing):
# Tears of Steel (live action, faces) and the Sintel trailer (CG animation).
set -euo pipefail
out=$1
mkdir -p "$out"
live=${2:-$out/src-tears_of_steel_720p.mov}
anim=${3:-$out/src-sintel_trailer-480p.mp4}
[ -f "$anim" ] || curl -sSfL -o "$anim" https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4
[ -f "$live" ] || curl -sSfL -o "$live" https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov
F="ffmpeg -v error -y"
T=${CORPUS_SECONDS:-1}
# Tears of Steel: two people on a bridge at 0:28, a face close-up at 0:52. Shorter sources
# (e.g. the Sintel trailer used for both on CI) use earlier points.
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$live" | cut -d. -f1)
if [ "${dur:-0}" -ge 120 ]; then A=28; B=52; else A=20; B=30; fi

# Live action.
$F -ss $A -t "$T" -i "$live" -c:v libx264 -crf 20 -preset fast -pix_fmt yuv420p -c:a aac -b:a 128k "$out/01 live action h264.mp4"
$F -ss $B -t "$T" -i "$live" -vf "scale=480:-2" -c:v libx264 -crf 33 -preset fast -c:a aac "$out/02 low quality faces 480w.mp4"
$F -ss $A -t "$T" -i "$live" -c:v libx265 -x265-params log-level=error -crf 24 -pix_fmt yuv420p10le -c:a libopus -b:a 96k "$out/03 hevc 10-bit.mkv"
$F -ss $A -t "$T" -i "$live" -vf "scale=640:-2" -c:v prores_ks -profile:v 2 -c:a pcm_s16le "$out/04 prores 422.mov"
$F -ss $A -t "$T" -i "$live" -vf "scale=640:-2" -c:v dnxhd -profile:v dnxhr_lb -pix_fmt yuv422p -c:a pcm_s16le -ar 48000 "$out/05 dnxhr.mxf" || true
# Animation.
$F -ss 10 -t "$T" -i "$anim" -c:v copy -c:a copy "$out/10 animation h264.mp4"
$F -ss 10 -t "$T" -i "$anim" -c:v libvpx-vp9 -b:v 800k -deadline realtime -cpu-used 8 -c:a libopus "$out/11 vp9 opus.webm"
$F -ss 10 -t "$T" -i "$anim" -c:v libsvtav1 -preset 12 -crf 40 -c:a libopus "$out/12 av1.mkv"
$F -ss 10 -t "$T" -i "$anim" -vf "scale=320:-2" -c:v mpeg4 -vtag xvid -q:v 6 -c:a libmp3lame "$out/13 xvid 320w.avi"
$F -ss 10 -t "$T" -i "$anim" -vf "scale=176:144" -c:v h263 -c:a aac -ar 8000 -ac 1 "$out/14 old phone 3gp.3gp"
# DVD-style interlaced MPEG-2 (top field first, 720x576, MP2 audio).
$F -ss $A -t "$T" -i "$live" -vf "scale=720:576,fps=50,tinterlace=interleave_top,setfield=tff" -c:v mpeg2video -b:v 6M -flags +ilme+ildct -top 1 -c:a mp2 -ar 48000 "$out/20 dvd interlaced.mpg"
# Phone portrait with rotation metadata (pixels landscape, shown upright by players).
$F -ss $A -t "$T" -i "$live" -vf "crop=300:534,scale=-2:1280,transpose=2" -c:v libx264 -crf 22 -c:a aac "$out/tmp_rot.mp4"
$F -display_rotation 90 -i "$out/tmp_rot.mp4" -c copy "$out/21 phone portrait rotated.mp4"
rm -f "$out/tmp_rot.mp4"
# Variable frame rate (timestamps jitter like a phone recording).
$F -ss $A -t "$T" -i "$live" -vf "setpts='N/(24*TB)+0.004*sin(N)/TB'" -fps_mode passthrough -c:v libx264 -crf 22 -c:a aac "$out/22 variable frame rate.mp4"
# Odd dimensions (lossless FFV1 allows them) and no audio.
$F -ss $A -t "$T" -i "$live" -vf "scale=641:267" -an -c:v ffv1 "$out/23 odd size no audio.mkv"
# Two audio tracks and a subtitle track.
printf '1\n00:00:00,000 --> 00:00:00,800\nHello\n' > "$out/sub.srt"
$F -ss $A -t "$T" -i "$live" -i "$out/sub.srt" -map 0:v -map 0:a -map 0:a -map 1 -c:v libx264 -crf 22 -c:a:0 aac -c:a:1 ac3 -c:s srt "$out/24 two audio tracks subs.mkv"
rm -f "$out/sub.srt"
# 60 fps, 4:2:2 10-bit, HDR10.
$F -ss $A -t "$T" -i "$live" -vf "fps=60" -c:v libx264 -crf 22 -c:a aac "$out/25 60fps.mp4"
$F -ss $A -t "$T" -i "$live" -c:v libx264 -profile:v high422 -pix_fmt yuv422p10le -crf 22 -c:a aac "$out/26 h264 422 10-bit.mov"
$F -ss $A -t "$T" -i "$live" -vf "zscale=tin=709:min=709:pin=709:rin=tv:t=linear:npl=100:p=709:m=709,format=gbrpf32le,zscale=p=2020:t=smpte2084:m=2020_ncl:r=tv:npl=203,format=yuv420p10le" \
  -c:v libx265 -x265-params "log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1000,400" \
  -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc -c:a aac "$out/27 hdr10 pq.mp4"
# Non-ASCII file name with spaces; very short clip (5 frames).
cp "$out/10 animation h264.mp4" "$out/30 Vidéo d'été – 日本語 テスト.mp4"
$F -ss $A -i "$live" -frames:v 5 -c:v libx264 -crf 22 -an "$out/31 five frames.mp4"
ls -la "$out"
