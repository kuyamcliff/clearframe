#!/usr/bin/env bash
# Builds the FFmpeg that ships inside the macOS app: static, GPL (redistributable, no
# "nonfree" parts), with vid.stab (stabilization), zimg (HDR conversion), x264, dav1d (AV1
# decoding), x265 (8- and 10-bit HEVC, needed for HDR10) and Apple's VideoToolbox hardware encoders. On macOS it builds arm64 and x86_64
# and joins them into universal binaries.
#
#   packaging/macos/build-ffmpeg.sh [output dir]      (default: apps/desktop/src-tauri/ffmpeg)
#
# Every source is pinned to an exact commit or checksum, and FFMPEG_SOURCE.txt in the output
# records them, so the build can be reproduced from upstream sources (GPL corresponding source).
# On Linux it builds for the host only, which is how the script is tested outside macOS.
set -euo pipefail

FFMPEG_VERSION=8.1
FFMPEG_SHA256=b072aed6871998cce9b36e7774033105ca29e33632be5b6347f3206898e0756a
X264_COMMIT=b35605ace3ddf7c1a5d67a2eb553f034aef41d55     # x264 "stable" branch
X265_TAG=4.1 X265_COMMIT=1d117bed4747758b51bd2c124d738527e30392cb
ZIMG_TAG=release-3.0.6 ZIMG_COMMIT=f819b14e8f39d1282400b0d9543e8ef73c1b2bbd
VIDSTAB_TAG=v1.1.1 VIDSTAB_COMMIT=90c76aca2cb06c3ff6f7476a7cd6851b39436656
DAV1D_TAG=1.5.4 DAV1D_COMMIT=54706fc6bc0cdecab7e9593974a4039cc038fca7

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(mkdir -p "${1:-$ROOT/apps/desktop/src-tauri/ffmpeg}" && cd "${1:-$ROOT/apps/desktop/src-tauri/ffmpeg}" && pwd)
WORK=${FFMPEG_BUILD_DIR:-$(mktemp -d)}
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
OS=$(uname -s)
HOST_ARCH=$(uname -m)
if [ "$OS" = Darwin ]; then
  ARCHS=${ARCHS:-"arm64 x86_64"}
  export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}
else
  ARCHS=$HOST_ARCH
fi
echo "Building FFmpeg $FFMPEG_VERSION for: $ARCHS (work dir $WORK)"
mkdir -p "$WORK/src"

clone() { # url tag-or-empty commit dir
  local dir="$WORK/src/$4"
  if [ ! -d "$dir/.git" ]; then
    git init -q "$dir"
    git -C "$dir" fetch -q --depth 1 "$1" "${2:-$3}"
    git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD
    # Keep the tag: some projects (x265) read their version from it.
    [ -n "$2" ] && git -C "$dir" tag -f "$2" FETCH_HEAD >/dev/null
  fi
  local got
  got=$(git -C "$dir" rev-parse HEAD)
  [ "$got" = "$3" ] || { echo "$4: expected commit $3, got $got" >&2; exit 1; }
}
clone https://code.videolan.org/videolan/x264.git "" "$X264_COMMIT" x264
clone https://bitbucket.org/multicoreware/x265_git.git "$X265_TAG" "$X265_COMMIT" x265
clone https://github.com/sekrit-twc/zimg.git "$ZIMG_TAG" "$ZIMG_COMMIT" zimg
clone https://github.com/georgmartius/vid.stab.git "$VIDSTAB_TAG" "$VIDSTAB_COMMIT" vidstab
clone https://code.videolan.org/videolan/dav1d.git "$DAV1D_TAG" "$DAV1D_COMMIT" dav1d
git -C "$WORK/src/zimg" submodule update -q --init --depth 1
# x265 4.1 asks for two pre-3.1 CMake policies that CMake 4 no longer supports. Use the new
# behaviour, under which Apple's compiler reports itself as "AppleClang", so match on "Clang".
perl -pi -e 's/cmake_policy\(SET (CMP0025|CMP0054) OLD\)/cmake_policy(SET $1 NEW)/; s/if\(\$\{CMAKE_CXX_COMPILER_ID\} STREQUAL "Clang"\)/if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")/' \
  "$WORK/src/x265/source/CMakeLists.txt"
if [ ! -d "$WORK/src/ffmpeg-$FFMPEG_VERSION" ]; then
  curl -sSLf -o "$WORK/src/ffmpeg.tar.xz" "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz"
  echo "$FFMPEG_SHA256  $WORK/src/ffmpeg.tar.xz" | shasum -a 256 -c -
  tar -xJf "$WORK/src/ffmpeg.tar.xz" -C "$WORK/src"
fi

for arch in $ARCHS; do
  P="$WORK/$arch"
  B="$WORK/build-$arch"
  rm -rf "$B" && mkdir -p "$B" "$P"
  # (macOS ships bash 3.2, where an empty array counts as unset under `set -u`, so arrays are
  # expanded as ${a[@]+"${a[@]}"} and cross-compiling is tracked in CROSS.)
  cross=() CROSS=0
  if [ "$OS" = Darwin ]; then
    triple=$([ "$arch" = arm64 ] && echo aarch64-apple-darwin || echo x86_64-apple-darwin)
    FLAGS="-arch $arch -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
    if [ "$arch" != "$HOST_ARCH" ]; then cross=(--host="$triple") CROSS=1; fi
    export CC=clang CXX=clang++
  else
    FLAGS="-fPIC"
  fi
  export CFLAGS="$FLAGS -O2" CXXFLAGS="$FLAGS -O2" LDFLAGS="$FLAGS"
  # Only ever link the libraries built here, never ones from Homebrew or the system.
  export PKG_CONFIG_PATH="$P/lib/pkgconfig" PKG_CONFIG_LIBDIR="$P/lib/pkgconfig"

  echo "== x264 ($arch)"
  cp -R "$WORK/src/x264" "$B/x264"
  (cd "$B/x264" && ./configure --prefix="$P" --enable-static --enable-pic --disable-cli --disable-opencl \
    ${cross[@]+"${cross[@]}"} --extra-cflags="$CFLAGS" --extra-ldflags="$LDFLAGS" >/dev/null && make -s -j"$JOBS" && make -s install)

  echo "== x265 ($arch, 8- and 10-bit)"
  # The 10-bit library is linked into the 8-bit one so a single libx265 encodes both.
  x265_opts=(-DCMAKE_BUILD_TYPE=Release -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_OSX_ARCHITECTURES="$arch" -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS="$CXXFLAGS")
  # Cross-compiled slices skip x265's hand-written assembly (its build picks it by host CPU).
  [ "$CROSS" = 1 ] && x265_opts+=(-DENABLE_ASSEMBLY=OFF -DCMAKE_SYSTEM_PROCESSOR="$arch")
  cmake -S "$WORK/src/x265/source" -B "$B/x265-10" "${x265_opts[@]}" -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF >/dev/null
  cmake --build "$B/x265-10" -j "$JOBS" >/dev/null
  cp "$B/x265-10/libx265.a" "$B/libx265_main10.a"
  cmake -S "$WORK/src/x265/source" -B "$B/x265-8" "${x265_opts[@]}" -DCMAKE_INSTALL_PREFIX="$P" -DCMAKE_INSTALL_LIBDIR=lib \
    -DEXTRA_LIB="$B/libx265_main10.a" -DLINKED_10BIT=ON >/dev/null
  cmake --build "$B/x265-8" -j "$JOBS" >/dev/null && cmake --install "$B/x265-8" >/dev/null
  if [ "$OS" = Darwin ]; then
    libtool -static -o "$P/lib/libx265.a" "$B/x265-8/libx265.a" "$B/libx265_main10.a"
  else
    printf 'CREATE %s\nADDLIB %s\nADDLIB %s\nSAVE\nEND\n' "$P/lib/libx265.a" "$B/x265-8/libx265.a" "$B/libx265_main10.a" | ar -M
  fi

  echo "== zimg ($arch)"
  cp -R "$WORK/src/zimg" "$B/zimg"
  (cd "$B/zimg" && ./autogen.sh >/dev/null 2>&1 && ./configure -q --prefix="$P" --disable-shared --enable-static ${cross[@]+"${cross[@]}"} \
    && make -s -j"$JOBS" && make -s install)

  echo "== vid.stab ($arch)"
  cmake -S "$WORK/src/vidstab" -B "$B/vidstab" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_SHARED_LIBS=OFF -DUSE_OMP=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_OSX_ARCHITECTURES="$arch" -DCMAKE_C_FLAGS="$CFLAGS" >/dev/null
  cmake --build "$B/vidstab" -j "$JOBS" >/dev/null && cmake --install "$B/vidstab" >/dev/null

  echo "== dav1d ($arch)"
  meson_cross=()
  if [ "$CROSS" = 1 ]; then
    cat > "$B/cross.txt" <<EOF
[binaries]
c = 'clang'
cpp = 'clang++'
ar = 'ar'
strip = 'strip'
nasm = 'nasm'
[built-in options]
c_args = ['-arch', '$arch', '-mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET']
c_link_args = ['-arch', '$arch', '-mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET']
[host_machine]
system = 'darwin'
cpu_family = '$([ "$arch" = arm64 ] && echo aarch64 || echo x86_64)'
cpu = '$arch'
endian = 'little'
EOF
    meson_cross=(--cross-file "$B/cross.txt")
  fi
  meson setup "$B/dav1d" "$WORK/src/dav1d" ${meson_cross[@]+"${meson_cross[@]}"} --prefix="$P" --libdir=lib --buildtype=release \
    --default-library=static -Denable_tools=false -Denable_tests=false >/dev/null
  ninja -C "$B/dav1d" install >/dev/null

  echo "== FFmpeg ($arch)"
  cp -R "$WORK/src/ffmpeg-$FFMPEG_VERSION" "$B/ffmpeg"
  conf=(--prefix="$P" --pkg-config-flags=--static --enable-gpl --disable-autodetect --disable-network
    --disable-ffplay --disable-doc --disable-debug
    --enable-libx264 --enable-libx265 --enable-libzimg --enable-libvidstab --enable-libdav1d
    --extra-cflags="$CFLAGS -I$P/include" --extra-ldflags="$LDFLAGS -L$P/lib")
  if [ "$OS" = Darwin ]; then
    conf+=(--enable-videotoolbox --enable-audiotoolbox --enable-zlib --enable-bzlib --enable-iconv --extra-libs="-lc++")
    [ "$CROSS" = 1 ] && conf+=(--enable-cross-compile --arch="$arch" --target-os=darwin --cc=clang --cxx=clang++)
  else
    conf+=(--extra-libs="-lstdc++ -lm -lpthread")
  fi
  (cd "$B/ffmpeg" && ./configure "${conf[@]}" >"$B/ffmpeg-configure.log" 2>&1 || { tail -40 "$B/ffmpeg/ffbuild/config.log"; exit 1; })
  (cd "$B/ffmpeg" && make -s -j"$JOBS" && make -s install)
done

# Join the architectures and check the result.
mkdir -p "$OUT"
for tool in ffmpeg ffprobe; do
  if [ "$OS" = Darwin ]; then
    lipo -create $(for a in $ARCHS; do echo "$WORK/$a/bin/$tool"; done) -output "$OUT/$tool"
    codesign --force -s - "$OUT/$tool"
    if otool -L "$OUT/$tool" | grep -E "/opt/homebrew|/usr/local" ; then echo "$tool links outside the system" >&2; exit 1; fi
    lipo -info "$OUT/$tool"
  else
    cp "$WORK/$HOST_ARCH/bin/$tool" "$OUT/$tool"
  fi
done
"$OUT/ffmpeg" -hide_banner -version | head -1
filters=$("$OUT/ffmpeg" -hide_banner -filters)
for f in vidstabdetect vidstabtransform zscale; do
  grep -q " $f " <<<"$filters" || { echo "missing filter $f" >&2; exit 1; }
done
encoders=$("$OUT/ffmpeg" -hide_banner -encoders)
for e in libx264 libx265; do grep -q " $e " <<<"$encoders" || { echo "missing encoder $e" >&2; exit 1; }; done
# HDR10 needs 10-bit HEVC.
"$OUT/ffmpeg" -v error -f lavfi -i testsrc2=size=64x64 -frames:v 2 -c:v libx265 -pix_fmt yuv420p10le -f null - \
  || { echo "libx265 cannot encode 10-bit" >&2; exit 1; }
grep -q libdav1d <<<"$("$OUT/ffmpeg" -hide_banner -decoders)" || { echo "missing libdav1d" >&2; exit 1; }
if grep -q -- --enable-nonfree <<<"$("$OUT/ffmpeg" -hide_banner -version)"; then echo "nonfree build" >&2; exit 1; fi
if [ "$OS" = Darwin ] && [ "$HOST_ARCH" = arm64 ] && arch -x86_64 /usr/bin/true 2>/dev/null; then
  arch -x86_64 "$OUT/ffmpeg" -hide_banner -version | head -1
fi

cp "$WORK/src/ffmpeg-$FFMPEG_VERSION/COPYING.GPLv2" "$OUT/FFMPEG_LICENSE.txt"
cat > "$OUT/FFMPEG_SOURCE.txt" <<EOF
FFmpeg $FFMPEG_VERSION, built by packaging/macos/build-ffmpeg.sh in the ClipFX Enhance repository
(https://github.com/kuyamcliff/clipfx-enhance) from these sources:

  FFmpeg   https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz (sha256 $FFMPEG_SHA256)
  x264     https://code.videolan.org/videolan/x264 commit $X264_COMMIT
  x265     https://bitbucket.org/multicoreware/x265_git $X265_TAG ($X265_COMMIT)
  zimg     https://github.com/sekrit-twc/zimg $ZIMG_TAG ($ZIMG_COMMIT)
  vid.stab https://github.com/georgmartius/vid.stab $VIDSTAB_TAG ($VIDSTAB_COMMIT)
  dav1d    https://code.videolan.org/videolan/dav1d $DAV1D_TAG ($DAV1D_COMMIT)

FFmpeg is licensed under the GNU GPL version 2 or later (see FFMPEG_LICENSE.txt) in this
configuration. Architectures: $ARCHS.

$("$OUT/ffmpeg" -hide_banner -version | grep '^configuration:')
EOF
echo "FFmpeg installed in $OUT"
