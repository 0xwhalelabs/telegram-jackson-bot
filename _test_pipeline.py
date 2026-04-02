"""Test the full animated sticker pipeline: download -> decrypt -> detect animation -> ffmpeg convert"""
import os
import subprocess
import tempfile
import urllib.request

# Step 1: Download one animated webp
item_code = "4448472"
url = f"https://item.kakaocdn.net/dw/{item_code}.emot_001.webp"
headers = {"User-Agent": "Android"}
req = urllib.request.Request(url, headers=headers)
resp = urllib.request.urlopen(req, timeout=10)
raw_data = resp.read()
print(f"Step 1: Downloaded {len(raw_data)} bytes")
print(f"  Magic (hex): {raw_data[:12].hex()}")
print(f"  Is RIFF/WEBP: {raw_data[:4] == b'RIFF' and b'WEBP' in raw_data[:12]}")

# Step 2: XOR decrypt
import sys
sys.path.insert(0, os.path.dirname(__file__))
from kakao_emoticon_bot.decrypt_kakao import DecryptKakao

decrypted = DecryptKakao.xor_data(raw_data)
print(f"\nStep 2: Decrypted {len(decrypted)} bytes")
print(f"  Magic (hex): {decrypted[:12].hex()}")
print(f"  Is RIFF/WEBP: {decrypted[:4] == b'RIFF' and b'WEBP' in decrypted[:12]}")
print(f"  Has ANIM: {b'ANIM' in decrypted[:200]}")
print(f"  Has ANMF: {b'ANMF' in decrypted[:2000]}")

# Step 3: Check with ffprobe
with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
    f.write(decrypted)
    tmp_webp = f.name

result = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,nb_frames,duration", 
     "-of", "json", tmp_webp],
    capture_output=True, text=True
)
print(f"\nStep 3: ffprobe output:")
print(result.stdout)
if result.stderr:
    print(f"  stderr: {result.stderr[:300]}")

# Step 4: Try ffmpeg conversion to webm
tmp_webm = tmp_webp.replace(".webp", ".webm")
cmd = [
    "ffmpeg", "-y",
    "-i", tmp_webp,
    "-vf", "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2:color=0x00000000",
    "-c:v", "libvpx-vp9",
    "-pix_fmt", "yuva420p",
    "-crf", "30",
    "-b:v", "0",
    "-t", "3",
    "-r", "30",
    "-an",
    "-auto-alt-ref", "0",
    tmp_webm,
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
print(f"\nStep 4: ffmpeg conversion")
print(f"  Return code: {result.returncode}")
if result.returncode == 0:
    size = os.path.getsize(tmp_webm)
    print(f"  Output size: {size} bytes ({size/1024:.1f} KB)")
    print(f"  Under 256KB: {size <= 256*1024}")
else:
    print(f"  stderr: {result.stderr[-500:]}")

# Step 5: Check is_animated_image function
def is_animated_image(data):
    if data[:4] == b"RIFF" and b"WEBP" in data[:12]:
        return b"ANIM" in data[:200]
    if data[:3] == b"GIF":
        return True
    return False

print(f"\nStep 5: is_animated_image check")
print(f"  Raw data: {is_animated_image(raw_data)}")
print(f"  Decrypted data: {is_animated_image(decrypted)}")

# Cleanup
os.unlink(tmp_webp)
if os.path.exists(tmp_webm):
    os.unlink(tmp_webm)
