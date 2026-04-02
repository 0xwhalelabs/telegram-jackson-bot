"""Check if emoji PNG files are APNG (animated) and test conversion."""
import urllib.request, os, subprocess, tempfile
from PIL import Image
from io import BytesIO

item_code = "1200327"

# Download emoji_001.png
url = f"https://item.kakaocdn.net/dw/{item_code}.emoji_001.png"
req = urllib.request.Request(url, headers={"User-Agent": "Android"})
raw = urllib.request.urlopen(req, timeout=10).read()
print(f"Downloaded: {len(raw)} bytes")
print(f"Magic: {raw[:8].hex()}")

# Check if APNG (look for acTL chunk)
is_apng = b"acTL" in raw
print(f"Is APNG: {is_apng}")

# Open with Pillow
img = Image.open(BytesIO(raw))
print(f"Format: {img.format}, Size: {img.size}, Frames: {getattr(img, 'n_frames', 1)}")
print(f"Mode: {img.mode}")

n_frames = getattr(img, "n_frames", 1)
if n_frames > 1:
    print(f"Duration: {img.info.get('duration', 'N/A')}ms")
    
    # Extract frames and convert to WebM at 100x100
    with tempfile.TemporaryDirectory() as tmpdir:
        durations = []
        for i in range(n_frames):
            img.seek(i)
            frame = img.copy().convert("RGBA").resize((100, 100))
            frame.save(os.path.join(tmpdir, f"frame_{i:04d}.png"), "PNG")
            durations.append(img.info.get("duration", 50))
        
        avg_dur = sum(durations) / len(durations) if durations else 100
        fps = min(1000.0 / avg_dur, 30) if avg_dur > 0 else 10
        print(f"Frames: {n_frames}, avg_dur: {avg_dur:.0f}ms, fps: {fps:.1f}")
        
        output = os.path.join(tmpdir, "output.webm")
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
            output,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            size = os.path.getsize(output)
            print(f"WebM: {size} bytes ({size/1024:.1f} KB), under 256KB: {size <= 256*1024}")
        else:
            print(f"ffmpeg error: {result.stderr[-300:]}")
else:
    print("Static image (not animated)")
    # For static, just resize to 100x100
    img = img.convert("RGBA").resize((100, 100))
    out = BytesIO()
    img.save(out, "PNG")
    print(f"Static 100x100 PNG: {len(out.getvalue())} bytes")
