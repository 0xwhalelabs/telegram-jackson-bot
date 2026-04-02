"""Try to extract item_code from e.kakao.com page HTML without auth."""
import urllib.request, re

url = "https://e.kakao.com/t/bbangbbangs-a-naughty-boy"
req = urllib.request.Request(url, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})
resp = urllib.request.urlopen(req, timeout=10)
html = resp.read().decode("utf-8")

# Look for item_code patterns in page HTML
for pattern_name, pattern in [
    ("itemCode", r'itemCode["\s:]+(\d+)'),
    ("item_code", r'item_code["\s:]+(\d+)'),
    ("/dw/NUM.", r'/dw/(\d+)\.'),
    ("store/emoticon/", r'store/emoticon/(\d+)'),
    ("emoticon/NUM", r'emoticon[/](\d{5,})'),
]:
    matches = re.findall(pattern, html)
    if matches:
        print(f"{pattern_name}: {matches[:5]}")

# Also try emoticon.kakao.com store page
print("\n=== Trying emoticon.kakao.com ===")
# Get hashedItemCode from e.kakao.com API
import json
api_url = "https://e.kakao.com/api/v1/items/t/bbangbbangs-a-naughty-boy"
req2 = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
resp2 = urllib.request.urlopen(req2, timeout=10)
data = json.loads(resp2.read().decode())
result = data.get("result", {})
hashed = result.get("hashedItemCode", "")
print(f"hashedItemCode: {hashed}")

# Try emoticon.kakao.com/items/{hash} page
store_url = f"https://emoticon.kakao.com/items/{hashed}?lang=ko"
req3 = urllib.request.Request(store_url, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})
try:
    resp3 = urllib.request.urlopen(req3, timeout=10)
    html2 = resp3.read().decode("utf-8")
    for pattern_name, pattern in [
        ("/dw/NUM.", r'/dw/(\d+)\.'),
        ("store/emoticon/", r'store/emoticon/(\d+)'),
        ("itemCode", r'itemCode["\s:]+(\d+)'),
    ]:
        matches = re.findall(pattern, html2)
        if matches:
            print(f"{pattern_name}: {matches[:5]}")
    
    if not any(re.findall(p, html2) for _, p in [("/dw/NUM.", r'/dw/(\d+)\.')]):
        # Print a snippet to check what's there
        idx = html2.find("dw/")
        if idx > -1:
            print(f"dw/ context: ...{html2[idx-20:idx+50]}...")
except Exception as e:
    print(f"Error: {e}")
