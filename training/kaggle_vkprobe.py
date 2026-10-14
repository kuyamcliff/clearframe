"""Find what the NVIDIA Vulkan driver needs inside a Kaggle GPU container (diagnostics only)."""
import os, subprocess
def sh(c):
    print(f'\n$ {c}', flush=True); p = subprocess.run(c, shell=True, capture_output=True, text=True); print((p.stdout + p.stderr)[-5000:], flush=True)
ver = subprocess.run('nvidia-smi --query-gpu=driver_version --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.split()[0]
L = '/usr/lib/x86_64-linux-gnu'
sh('cat /proc/driver/nvidia/version; ls -la /dev/nvidia* /dev/dri 2>&1; env | grep -i nvidia')
sh('DEBIAN_FRONTEND=noninteractive apt-get update -qq && apt-get install -y -qq libvulkan1 vulkan-tools libx11-6 libxext6 libglvnd0 libegl1 libgl1 >/dev/null; echo ok')
sh(f'curl -sSfL -o /tmp/nv.run https://us.download.nvidia.com/tesla/{ver}/NVIDIA-Linux-x86_64-{ver}.run && sh /tmp/nv.run --extract-only --target /opt/nv >/dev/null; ls /opt/nv')
sh(f'cd /opt/nv && for f in libGLX_nvidia.so.{ver} libnvidia-*.so.{ver} libEGL_nvidia.so.{ver}; do [ -e {L}/$f ] || cp $f {L}/; done; ln -sf {L}/libGLX_nvidia.so.{ver} {L}/libGLX_nvidia.so.0; ldconfig 2>/dev/null; cat nvidia_icd.json*')
sh(f'mkdir -p /etc/vulkan/icd.d; cp /opt/nv/nvidia_icd.json /etc/vulkan/icd.d/ 2>/dev/null || sed "s#__NV_VK_ICD__#libGLX_nvidia.so.0#" /opt/nv/nvidia_icd.json.template > /etc/vulkan/icd.d/nvidia_icd.json; cat /etc/vulkan/icd.d/nvidia_icd.json')
sh('mkdir -p /usr/share/glvnd/egl_vendor.d && cp /opt/nv/10_nvidia.json /usr/share/glvnd/egl_vendor.d/ && cat /usr/share/glvnd/egl_vendor.d/10_nvidia.json; ls -la /usr/lib/x86_64-linux-gnu/libEGL_nvidia*')
sh('mknod -m 666 /dev/nvidia-modeset c 195 254 && ls -la /dev/nvidia-modeset || echo "mknod not permitted"')
sh('VK_LOADER_DEBUG=error,warn vulkaninfo --summary 2>&1 | tail -40')
sh('which strace >/dev/null || apt-get install -y -qq strace >/dev/null; strace -f -e trace=openat,open,ioctl vulkaninfo --summary 2>&1 | grep -E "/dev/|/proc/driver|EACCES|EPERM" | grep -v "libEGL_mesa\|gallium" | tail -60')
