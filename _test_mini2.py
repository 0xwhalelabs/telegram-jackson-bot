"""Test downloading mini emoticon animated files."""
import json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from kakao_emoticon_bot.decrypt_kakao import DecryptKakao
from PIL import Image
from io import BytesIO

item_code = "1200327"

# Try downloading first mini emoticon file
for i in range(1, 3):
    url = f"https://item.kakaocdn.net/dw/{item_code}.emot_{i:03d}.webp"
    req = urllib.request.Request(url, headers={"User-Agent": "Android"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        raw = resp.read()
        print(f"emot_{i:03d}.webp: {len(raw)} bytes, magic={raw[:4].hex()}")
        
        # Decrypt
        decrypted = DecryptKakao.xor_data(raw)
        print(f"  Decrypted magic: {decrypted[:12].hex()}")
        is_riff = decrypted[:4] == b"RIFF" and b"WEBP" in decrypted[:12]
        has_anim = b"ANIM" in decrypted[:200]
        print(f"  Is RIFF/WEBP: {is_riff}, Has ANIM: {has_anim}")
        
        # Open with Pillow
        img = Image.open(BytesIO(decrypted))
        print(f"  Size: {img.size}, Frames: {getattr(img, 'n_frames', 1)}")
        dur = img.info.get("duration", "N/A")
        print(f"  Duration per frame: {dur}ms")
    except urllib.error.HTTPError as e:
        print(f"emot_{i:03d}.webp: HTTP {e.code}")
    except Exception as e:
        print(f"emot_{i:03d}.webp: Error {e}")

# Also check if there's a different URL pattern for mini/emoji
# Try "emoj" prefix
for prefix in ["emoj", "emoji", "mini"]:
    url = f"https://item.kakaocdn.net/dw/{item_code}.{prefix}_001.webp"
    req = urllib.request.Request(url, headers={"User-Agent": "Android"})
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        raw = resp.read()
        print(f"\n{prefix}_001.webp: {len(raw)} bytes")
    except urllib.error.HTTPError as e:
        print(f"{prefix}_001.webp: HTTP {e.code}")
    except Exception as e:
        print(f"{prefix}_001.webp: Error {e}")
