import urllib.request
import json

# Search for 잔망 루피 16
query = urllib.parse.quote("잔망 루피 16")
import urllib.parse
url = f"https://e.kakao.com/api/v1/search?query={query}&page=0&size=5"
headers = {"User-Agent": "Mozilla/5.0"}

req = urllib.request.Request(url, headers=headers)
resp = urllib.request.urlopen(req, timeout=10)
data = json.loads(resp.read())

results = data.get("result", {}).get("content", [])
for r in results:
    title = r.get("title", "")
    title_url = r.get("titleUrl", "")
    print(f"title={title}, titleUrl={title_url}")
    
    # Now get detail
    detail_url = f"https://e.kakao.com/api/v1/items/t/{title_url}"
    req2 = urllib.request.Request(detail_url, headers=headers)
    try:
        resp2 = urllib.request.urlopen(req2, timeout=10)
        detail = json.loads(resp2.read())
        result = detail.get("result", {})
        print(f"  hashedItemCode={result.get('hashedItemCode')}")
        print(f"  itemCode={result.get('itemCode')}")
        print(f"  contentType={result.get('contentType')}")
        # Check if there's an eid or code in the data
        for k, v in result.items():
            if 'code' in k.lower() or 'id' in k.lower() or 'item' in k.lower():
                print(f"  {k}={v}")
    except Exception as e:
        print(f"  detail error: {e}")
    print()
