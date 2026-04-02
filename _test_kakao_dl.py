import urllib.request

item_code = "4448472"
headers = {"User-Agent": "Android"}

# Test file_pack.zip
try:
    req = urllib.request.Request(f"http://item.kakaocdn.net/dw/{item_code}.file_pack.zip", headers=headers)
    resp = urllib.request.urlopen(req, timeout=10)
    print(f"file_pack.zip: OK {len(resp.read())} bytes")
except Exception as e:
    print(f"file_pack.zip: FAIL {e}")

# Test individual files
for ext in ["webp", "gif", "png"]:
    for play_type in ["emot", "emoji", ""]:
        prefix = f"{play_type}_" if play_type else ""
        url = f"https://item.kakaocdn.net/dw/{item_code}.{prefix}001.{ext}"
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=5)
            data = resp.read()
            magic = data[:4].hex()
            print(f"{prefix}001.{ext}: OK {len(data)} bytes, magic={magic}")
        except Exception as e:
            print(f"{prefix}001.{ext}: FAIL")

# Count how many emot_XXX.webp exist
print("\nCounting emot_XXX.webp files:")
count = 0
for i in range(1, 33):
    url = f"https://item.kakaocdn.net/dw/{item_code}.emot_{i:03d}.webp"
    try:
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=5)
        data = resp.read()
        count += 1
        print(f"  emot_{i:03d}.webp: {len(data)} bytes")
    except:
        print(f"  emot_{i:03d}.webp: FAIL (stopped)")
        break
print(f"Total: {count} files found")
