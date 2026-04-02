"""Test: Pillow frame extraction -> ffmpeg WebM conversion"""
import os, subprocess, tempfile, urllib.request
from PIL import Image
from io import BytesIO

# Download + decrypt
item_code = "4448472"
url = f"https://item.kakaocdn.net/dw/{item_code}.emot_001.webp"
req = urllib.request.Request(url, headers={"User-Agent": "Android"})
raw = urllib.request.urlopen(req, timeout=10).read()

import sys; sys.path.insert(0, os.path.dirname(__file__))
from kakao_emoticon_bot.decrypt_kakao import DecryptKakao
decrypted = DecryptKakao.xor_data(raw)

# Open with Pillow
img = Image.open(BytesIO(decrypted))
print(f"Format: {img.format}, Size: {img.size}, Frames: {getattr(img, 'n_frames', 1)}")
print(f"Info keys: {list(img.info.keys())}")

n_frames = getattr(img, "n_frames", 1)
if n_frames <= 1:
    print("NOT ANIMATED - only 1 frame")
    exit(1)

# Extract frames to temp dir
with tempfile.TemporaryDirectory() as tmpdir:
    durations = []
    for i in range(n_frames):
        img.seek(i)
        frame = img.copy().convert("RGBA").resize((512, 512))
        frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
        frame.save(frame_path, "PNG")
        dur = img.info.get("duration", 50)
        durations.append(dur)
    
    avg_dur = sum(durations) / len(durations)
    fps = 1000.0 / avg_dur if avg_dur > 0 else 30
    fps = min(fps, 30)  # Telegram max 30fps
    
    print(f"Extracted {n_frames} frames, avg duration={avg_dur:.0f}ms, fps={fps:.1f}")
    
    # ffmpeg: frames -> webm
    output_webm = os.path.join(tmpdir, "output.webm")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmpdir, "frame_%04d.png"),
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-crf", "30",
        "-b:v", "0",
        "-t", "3",
        "-an",
        "-auto-alt-ref", "0",
        output_webm,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    print(f"\nffmpeg return code: {result.returncode}")
    if result.returncode == 0:
        size = os.path.getsize(output_webm)
        print(f"Output: {size} bytes ({size/1024:.1f} KB)")
        print(f"Under 256KB: {size <= 256*1024}")
    else:
        print(f"stderr: {result.stderr[-500:]}")
    
    # Try higher CRF if too big
    if result.returncode == 0 and os.path.getsize(output_webm) > 256*1024:
        for crf in [40, 50, 63]:
            cmd[cmd.index("-crf")+1] = str(crf)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                size = os.path.getsize(output_webm)
                print(f"CRF {crf}: {size} bytes ({size/1024:.1f} KB), under 256KB: {size <= 256*1024}")
                if size <= 256*1024:
                    break
