"""Explore mini emoticon download patterns more thoroughly."""
import json, urllib.request, urllib.parse

with open(".kakao_auth_tokens.json") as f:
    token = list(json.load(f).values())[0]

item_code = "1200327"

# Try various URL patterns
patterns = [
    f"https://item.kakaocdn.net/dw/{item_code}.emot_001.webp",
    f"https://item.kakaocdn.net/dw/{item_code}.emot_001.png",
    f"https://item.kakaocdn.net/dw/{item_code}.emot_001.gif",
    f"https://item.kakaocdn.net/dw/{item_code}.sound_001.mp3",
    f"https://item.kakaocdn.net/dw/{item_code}.file_pack.zip",
    f"https://item.kakaocdn.net/dw/{item_code}.title.webp",
    f"https://item.kakaocdn.net/dw/{item_code}.title.png",
    f"https://item.kakaocdn.net/dw/{item_code}.thumb_001.png",
    f"https://item.kakaocdn.net/dw/{item_code}.thumb_001.webp",
    f"https://item.kakaocdn.net/dw/{item_code}.preview.webp",
    f"https://item.kakaocdn.net/dw/{item_code}.preview.gif",
]

for url in patterns:
    req = urllib.request.Request(url, headers={"User-Agent": "Android"})
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        raw = resp.read()
        suffix = url.split(item_code)[-1]
        print(f"OK {suffix}: {len(raw)} bytes, magic={raw[:4].hex()}")
    except urllib.error.HTTPError as e:
        suffix = url.split(item_code)[-1]
        print(f"   {suffix}: HTTP {e.code}")
    except Exception as e:
        suffix = url.split(item_code)[-1]
        print(f"   {suffix}: {e}")

# Try Kakao API to get download info
print("\n=== Trying Kakao internal API ===")
# item info API
api_url = f"https://talk-pilsner.kakao.com/emoticon/api/store/v3/items/{item_code}"
req = urllib.request.Request(api_url, headers={"Authorization": token, "User-Agent": "Android"})
try:
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read().decode())
    print(f"Item info: {json.dumps(data, indent=2, ensure_ascii=False)[:2000]}")
except Exception as e:
    print(f"Item info error: {e}")

# Try download API
print("\n=== Trying download API ===")
for endpoint in [
    f"https://talk-pilsner.kakao.com/emoticon/api/store/v3/items/{item_code}/download",
    f"https://talk-pilsner.kakao.com/emoticon/api/v3/items/{item_code}/download",
    f"https://talk-pilsner.kakao.com/emoticon/item/{item_code}",
]:
    req = urllib.request.Request(endpoint, headers={"Authorization": token, "User-Agent": "Android"})
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        data = resp.read().decode()[:500]
        print(f"OK {endpoint.split('/')[-2:]}: {data}")
    except Exception as e:
        print(f"   {'/'.join(endpoint.split('/')[-2:])}: {e}")
