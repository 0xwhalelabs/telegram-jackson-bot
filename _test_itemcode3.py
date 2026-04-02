import urllib.request
import json
import re

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Step 1: Get detail for zanmang-loopy-ver16
title_url = "zanmang-loopy-ver16"
detail_url = f"https://e.kakao.com/api/v1/items/t/{title_url}"
req = urllib.request.Request(detail_url, headers=headers)
resp = urllib.request.urlopen(req, timeout=10)
data = json.loads(resp.read())
result = data.get("result", {})

print("=== All fields ===")
for k, v in result.items():
    if isinstance(v, (str, int, bool, type(None))):
        print(f"  {k} = {v}")
    elif isinstance(v, list) and len(v) < 5:
        print(f"  {k} = {v}")
    elif isinstance(v, list):
        print(f"  {k} = [{len(v)} items]")
    elif isinstance(v, dict):
        print(f"  {k} = {{...}}")

hashed = result.get("hashedItemCode", "")
print(f"\nhashedItemCode: '{hashed}'")

# Step 2: If we have hashedItemCode, try share page
if hashed:
    share_url = f"https://emoticon.kakao.com/items/{hashed}"
    print(f"\nTrying share URL: {share_url}")
    req2 = urllib.request.Request(share_url, headers=headers)
    resp2 = urllib.request.urlopen(req2, timeout=10)
    html = resp2.read().decode("utf-8", errors="ignore")
    
    # Look for kakaotalk://store/emoticon/{eid}
    matches = re.findall(r'kakaotalk://store/emoticon/(\d+)', html)
    print(f"eid from kakaotalk:// scheme: {matches}")
    
    matches2 = re.findall(r'"itemCode"\s*:\s*"?(\d+)"?', html)
    print(f"itemCode from JSON: {matches2}")
    
    matches3 = re.findall(r'"contentId"\s*:\s*"?(\d+)"?', html)
    print(f"contentId from JSON: {matches3}")
    
    # Save first 5000 chars for inspection
    print(f"\nHTML snippet (first 3000 chars):")
    print(html[:3000])
