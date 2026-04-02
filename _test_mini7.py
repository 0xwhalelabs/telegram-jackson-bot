"""Check if mini emoticons have animated webp files via authed API."""
import json, sys, os, urllib.request
sys.path.insert(0, '.')
from kakao_emoticon_bot.message import _extract_token_from_desktop
from kakao_emoticon_bot.decrypt_kakao import DecryptKakao
from PIL import Image
from io import BytesIO

token = _extract_token_from_desktop()
print("token_len:", len(token) if token else 0)

if not token:
    tokens = json.load(open('.kakao_auth_tokens.json'))
    token = list(tokens.values())[0]

headers_authed = {
    "Authorization": token,
    "Talk-Agent": "android/10.8.1",
    "Talk-Language": "en",
    "User-Agent": "okhttp/4.10.0",
}

# Get pack info for item_code 1200327 (mini-i-ticon)
item_code = "1200327"
req = urllib.request.Request(
    f"https://talk-pilsner.kakao.com/emoticon/api/store/v3/items/{item_code}",
    headers=headers_authed,
    method="POST"
)
try:
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read().decode())
    unit = data.get("itemUnitInfo", [{}])[0]
    preview = unit.get("previewData", {})
    ppf = preview.get("playPathFormat", "")
    pf = preview.get("pathFormat", "")
    spf = preview.get("soundPathFormat", "")
    num = preview.get("num", 0)
    subtype = unit.get("itemSubtype", "?")
    print(f"item_code={item_code}, subtype={subtype}, num={num}")
    print(f"  pathFormat: {pf}")
    print(f"  playPathFormat: {ppf}")
    print(f"  soundPathFormat: {spf}")

    # Download using playPathFormat
    if ppf:
        dl_url = "https://item.kakaocdn.net/" + ppf.replace("##", "01")
        print(f"\nDownloading: {dl_url}")
        req2 = urllib.request.Request(dl_url, headers={"User-Agent": "Android"})
        raw = urllib.request.urlopen(req2, timeout=10).read()
        print(f"  Size: {len(raw)} bytes")
        
        # Check if needs decryption
        ext = ppf.split(".")[-1]
        if ext in ("webp", "gif"):
            decrypted = DecryptKakao.xor_data(raw)
            print(f"  Decrypted magic: {decrypted[:12].hex()}")
            img = Image.open(BytesIO(decrypted))
        else:
            img = Image.open(BytesIO(raw))
        
        print(f"  Format: {img.format}, Size: {img.size}, Frames: {getattr(img, 'n_frames', 1)}")
        is_apng = b"acTL" in raw
        print(f"  Is APNG: {is_apng}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
