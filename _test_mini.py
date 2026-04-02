"""Test Kakao mini emoticon download patterns."""
import json
import urllib.request
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Load auth token
with open(".kakao_auth_tokens.json") as f:
    token = list(json.load(f).values())[0]

# First, search for a mini emoticon pack to understand the structure
# Mini emoticons are typically labeled as "미니" in Kakao store
search_url = "https://e.kakao.com/api/v1/search?query=%EB%AF%B8%EB%8B%88&page=0&size=10"
req = urllib.request.Request(search_url, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})
resp = urllib.request.urlopen(req, timeout=10)
data = json.loads(resp.read().decode())

items = data.get("result", {}).get("content", [])
print(f"Found {len(items)} items:")
for item in items[:5]:
    title = item.get("title", "?")
    title_url = item.get("titleUrl", "")
    code = item.get("code", "")
    print(f"  - {title} | titleUrl={title_url} | code={code}")

# Get detail of first mini emoticon
if items:
    title_url = items[0]["titleUrl"]
    detail_url = f"https://e.kakao.com/api/v1/items/t/{title_url}"
    req = urllib.request.Request(detail_url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    resp = urllib.request.urlopen(req, timeout=10)
    detail = json.loads(resp.read().decode())
    result = detail.get("result", {})
    
    print(f"\n=== Detail for '{result.get('title')}' ===")
    print(f"  hashedItemCode: {result.get('hashedItemCode')}")
    print(f"  thumbnailUrls count: {len(result.get('thumbnailUrls', []))}")
    print(f"  type: {result.get('type')}")
    print(f"  category: {result.get('category')}")
    
    # Check all keys
    for k in sorted(result.keys()):
        v = result[k]
        if isinstance(v, (str, int, bool)):
            print(f"  {k}: {v}")
        elif isinstance(v, list):
            print(f"  {k}: [{len(v)} items]")
    
    # Try to get item_code via hash
    hashed = result.get("hashedItemCode", "")
    if hashed and token:
        import urllib.parse
        data_enc = urllib.parse.urlencode({"hashedItemCode": hashed}).encode()
        req = urllib.request.Request(
            "https://talk-pilsner.kakao.com/emoticon/api/store/v3/item-code-by-hash",
            data=data_enc,
            headers={"Authorization": token}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            print(f"\n  item_code response: {resp.read().decode()}")
        except Exception as e:
            print(f"\n  item_code error: {e}")
    
    # Try downloading mini emoticon files
    # Mini emoticons might use different URL patterns
    # Standard: https://item.kakaocdn.net/dw/{item_code}.emot_001.webp
    # Mini might use: .emot_001.webp with smaller size, or different prefix
    
    # Also check thumbnailUrls
    thumbs = result.get("thumbnailUrls", [])
    if thumbs:
        print(f"\n  First thumbnail: {thumbs[0]}")
        print(f"  Last thumbnail: {thumbs[-1]}")
