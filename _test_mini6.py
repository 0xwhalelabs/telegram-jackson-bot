"""Try more patterns for animated mini emoticon files."""
import json, urllib.request

with open(".kakao_auth_tokens.json") as f:
    token = list(json.load(f).values())[0]

item_code = "1200327"

# From API: pathFormat = "dw/1200327.thum_0##.png", playPathFormat = "dw/1200327.emoji_0##.png"
# The ## is replaced with numbers. Let's try various file extensions and patterns

base = f"https://item.kakaocdn.net/dw/{item_code}"

# Try webp versions of emoji files (might be encrypted animated)
patterns = [
    f"{base}.emoji_001.webp",
    f"{base}.emoji_001.gif",
    f"{base}.emoji_001.mp4",
    f"{base}.emoji_001.webm",
    # Try with auth
]

for url in patterns:
    suffix = url.split(item_code)[-1]
    for hdr_name, headers in [
        ("noauth", {"User-Agent": "Android"}),
        ("auth", {"User-Agent": "Android", "Authorization": token}),
    ]:
        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            raw = resp.read()
            print(f"OK [{hdr_name}] {suffix}: {len(raw)} bytes, magic={raw[:8].hex()}")
        except urllib.error.HTTPError as e:
            print(f"   [{hdr_name}] {suffix}: HTTP {e.code}")
        except Exception as e:
            print(f"   [{hdr_name}] {suffix}: {e}")

# Check sticker-convert source for emoji download patterns
print("\n=== Checking if emoji type uses different download mechanism ===")

# Try the pilsner API for emoji-specific download info
for endpoint in [
    f"https://talk-pilsner.kakao.com/emoticon/api/store/v3/emoji/{item_code}",
    f"https://talk-pilsner.kakao.com/emoticon/api/store/v3/emoji/{item_code}/items",
    f"https://talk-pilsner.kakao.com/emoticon/api/v3/emoji/{item_code}",
]:
    req = urllib.request.Request(endpoint, headers={"Authorization": token, "User-Agent": "Android"})
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        data = resp.read().decode()[:500]
        print(f"OK {endpoint.split('v3/')[-1]}: {data}")
    except Exception as e:
        print(f"   {endpoint.split('v3/')[-1]}: {e}")

# The PNG files from playPathFormat might actually be the final format
# Telegram custom emoji supports: static (100x100 PNG) and video (100x100 WebM)
# If Kakao mini emoticons are static PNGs, we can still make them as static custom emoji
print("\n=== Checking multiple emoji PNGs for animation ===")
for i in [1, 2, 3]:
    url = f"{base}.emoji_{i:03d}.png"
    req = urllib.request.Request(url, headers={"User-Agent": "Android"})
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        raw = resp.read()
        is_apng = b"acTL" in raw
        print(f"emoji_{i:03d}.png: {len(raw)} bytes, APNG={is_apng}")
    except Exception as e:
        print(f"emoji_{i:03d}.png: {e}")

# Also check thum files
for i in [1, 2, 3]:
    url = f"{base}.thum_{i:03d}.png"
    req = urllib.request.Request(url, headers={"User-Agent": "Android"})
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        raw = resp.read()
        print(f"thum_{i:03d}.png: {len(raw)} bytes")
    except Exception as e:
        print(f"thum_{i:03d}.png: {e}")

# Try a known animated mini emoticon pack
# Search for "움직이는 미니"
print("\n=== Searching for animated mini emoticons ===")
search_url = "https://e.kakao.com/api/v1/search?query=%EC%9B%80%EC%A7%81%EC%9D%B4%EB%8A%94+%EB%AF%B8%EB%8B%88&page=0&size=5"
req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
resp = urllib.request.urlopen(req, timeout=10)
data = json.loads(resp.read().decode())
items = data.get("result", {}).get("content", [])
for item in items[:3]:
    print(f"  {item.get('title')} | {item.get('titleUrl')}")
