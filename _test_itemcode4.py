import urllib.request
import json
import re

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

hashed = "3fXxMs-10wXEWmClWkKRw6X9s2k="

# Method 1: Try emoticon.kakao.com API
urls_to_try = [
    f"https://emoticon.kakao.com/api/item/{hashed}",
    f"https://emoticon.kakao.com/api/items/{hashed}",
    f"https://emoticon.kakao.com/api/v1/items/{hashed}",
    f"https://e.kakao.com/api/v1/items/{hashed}",
]

for url in urls_to_try:
    try:
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=10)
        data = resp.read().decode("utf-8")
        print(f"OK {url}")
        print(f"  {data[:500]}")
    except Exception as e:
        print(f"FAIL {url}: {e}")

# Method 2: Try share page with mobile user agent
print("\n=== Mobile UA share page ===")
mobile_headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 11) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"}
share_url = f"https://emoticon.kakao.com/items/{hashed}"
try:
    req = urllib.request.Request(share_url, headers=mobile_headers)
    resp = urllib.request.urlopen(req, timeout=10)
    html = resp.read().decode("utf-8", errors="ignore")
    
    # Look for any numeric IDs
    matches = re.findall(r'kakaotalk://store/emoticon/(\d+)', html)
    print(f"kakaotalk scheme: {matches}")
    
    # Look in script tags for data
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
    for s in scripts:
        if 'itemCode' in s or 'contentId' in s or 'emoticon' in s.lower():
            print(f"Interesting script: {s[:500]}")
    
    # Look for __NEXT_DATA__ or similar
    next_data = re.findall(r'__NEXT_DATA__\s*=\s*({.*?})\s*</script>', html, re.DOTALL)
    if next_data:
        print(f"NEXT_DATA found: {next_data[0][:500]}")
    
    # Look for any JSON-like data with numbers
    all_numbers = re.findall(r'"(?:id|code|eid|itemCode|contentId)"\s*:\s*(\d+)', html)
    print(f"All numeric IDs: {all_numbers}")
    
except Exception as e:
    print(f"Error: {e}")

# Method 3: Try Kakao CDN brute-force approach - check if we can find item_code
# from the titleImageUrl hash
print("\n=== Try direct CDN with hash-based approach ===")
# The titleImageUrl contains a hash, maybe we can derive item_code from it
# Actually let's try the talk-pilsner API without auth
print("Trying talk-pilsner without auth...")
import urllib.parse
try:
    data = urllib.parse.urlencode({"hashedItemCode": hashed}).encode()
    req = urllib.request.Request(
        "https://talk-pilsner.kakao.com/emoticon/api/store/v3/item-code-by-hash",
        data=data,
        headers=headers
    )
    resp = urllib.request.urlopen(req, timeout=10)
    print(f"OK: {resp.read().decode()}")
except Exception as e:
    print(f"FAIL: {e}")
