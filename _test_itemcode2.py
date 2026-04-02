import urllib.request
import json
import re

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Method 1: Get detail for a known titleUrl
title_url = "pretty-ruffy-16"
detail_url = f"https://e.kakao.com/api/v1/items/t/{title_url}"
try:
    req = urllib.request.Request(detail_url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
    result = data.get("result", {})
    print("=== Detail API ===")
    for k, v in result.items():
        if isinstance(v, (str, int, bool)):
            print(f"  {k} = {v}")
except Exception as e:
    print(f"Detail API error: {e}")

# Method 2: Try share page HTML for kakaotalk://store/emoticon/{eid}
# First we need the hashedItemCode - let's try the share link
print("\n=== Share page HTML ===")
# Try known share URLs or construct from detail
try:
    req = urllib.request.Request(detail_url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
    result = data.get("result", {})
    hashed = result.get("hashedItemCode", "")
    print(f"hashedItemCode from API: '{hashed}'")
    
    if hashed:
        share_url = f"https://emoticon.kakao.com/items/{hashed}"
        req2 = urllib.request.Request(share_url, headers=headers)
        resp2 = urllib.request.urlopen(req2, timeout=10)
        html = resp2.read().decode("utf-8", errors="ignore")
        
        # Look for kakaotalk://store/emoticon/{eid}
        matches = re.findall(r'kakaotalk://store/emoticon/(\d+)', html)
        print(f"eid from HTML: {matches}")
        
        # Also look for any numeric IDs
        matches2 = re.findall(r'"itemCode"\s*:\s*"?(\d+)"?', html)
        print(f"itemCode from HTML: {matches2}")
        
        matches3 = re.findall(r'"eid"\s*:\s*"?(\d+)"?', html)
        print(f"eid from HTML json: {matches3}")
except Exception as e:
    print(f"Error: {e}")

# Method 3: Try searching and see what fields we get
print("\n=== Search API ===")
import urllib.parse
query = urllib.parse.quote("잔망 루피 16")
search_url = f"https://e.kakao.com/api/v1/search?query={query}&page=0&size=5"
try:
    req = urllib.request.Request(search_url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
    results = data.get("result", {}).get("content", [])
    for r in results:
        title = r.get("title", "")
        title_url = r.get("titleUrl", "")
        if "루피" in title:
            print(f"Found: title={title}, titleUrl={title_url}")
            for k, v in r.items():
                print(f"  {k} = {v}")
except Exception as e:
    print(f"Search error: {e}")

# Method 4: Try the detail for the found titleUrl
print("\n=== Detail for found titleUrl ===")
for tu in ["pretty-ruffy-16", "prettyruffy16", "잔망루피16"]:
    detail_url = f"https://e.kakao.com/api/v1/items/t/{urllib.parse.quote(tu)}"
    try:
        req = urllib.request.Request(detail_url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        result = data.get("result", {})
        print(f"\ntitleUrl={tu}: OK")
        for k, v in result.items():
            if isinstance(v, (str, int, bool)):
                print(f"  {k} = {v}")
    except Exception as e:
        print(f"titleUrl={tu}: {e}")
