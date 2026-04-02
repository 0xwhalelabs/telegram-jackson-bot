"""Explore mini emoticon file patterns - try emoji-specific URLs."""
import json, urllib.request

with open(".kakao_auth_tokens.json") as f:
    token = list(json.load(f).values())[0]

item_code = "1200327"

# Get full item info (print more)
api_url = f"https://talk-pilsner.kakao.com/emoticon/api/store/v3/items/{item_code}"
req = urllib.request.Request(api_url, headers={"Authorization": token, "User-Agent": "Android"})
resp = urllib.request.urlopen(req, timeout=10)
data = json.loads(resp.read().decode())
print(json.dumps(data, indent=2, ensure_ascii=False)[:5000])

# Try more URL patterns specific to emoji type
print("\n=== Testing emoji-specific URL patterns ===")
patterns = []
for ext in ["webp", "png", "gif"]:
    for prefix in ["emoj", "emoji", "emot", "img", "image", "thumb"]:
        for fmt in [f".{prefix}_001.{ext}", f".{prefix}001.{ext}"]:
            patterns.append(f"https://item.kakaocdn.net/dw/{item_code}{fmt}")

# Also try numbered patterns without prefix
for ext in ["webp", "png", "gif"]:
    patterns.append(f"https://item.kakaocdn.net/dw/{item_code}.001.{ext}")
    patterns.append(f"https://item.kakaocdn.net/dw/{item_code}.1.{ext}")

# Try with auth header
for url in patterns:
    suffix = url.split(item_code)[-1]
    for headers in [
        {"User-Agent": "Android"},
        {"User-Agent": "Android", "Authorization": token},
    ]:
        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=3)
            raw = resp.read()
            auth = "auth" if "Authorization" in headers else "noauth"
            print(f"OK [{auth}] {suffix}: {len(raw)} bytes")
            break
        except urllib.error.HTTPError:
            pass
        except:
            pass
