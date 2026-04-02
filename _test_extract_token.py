from sticker_convert.utils.process import find_pid_by_name
from PyMemoryEditor import OpenProcess

kakao_pid = find_pid_by_name("kakaotalk")
print(f"KakaoTalk PID: {kakao_pid}")

if kakao_pid is None:
    print("KakaoTalk not running")
    exit(1)

auth_token = None
try:
    with OpenProcess(pid=int(kakao_pid)) as process:
        print("Searching memory for 'authorization: '...")
        count = 0
        for address in process.search_by_value(str, 15, "authorization: "):
            count += 1
            auth_token_addr = address + 15
            auth_token_bytes = process.read_process_memory(auth_token_addr, bytes, 200)
            auth_token_term = auth_token_bytes.find(b"\x00")
            if auth_token_term == -1:
                continue
            candidate = auth_token_bytes[:auth_token_term].decode("ascii", errors="ignore")
            print(f"  Found candidate (len={len(candidate)}): {candidate[:30]}...")
            if len(candidate) > 150:
                auth_token = candidate
                break
        print(f"Searched {count} addresses")
except PermissionError as e:
    print(f"PermissionError: {e}")
except Exception as e:
    print(f"Error: {e}")

if auth_token:
    print(f"\nSUCCESS! Token length: {len(auth_token)}")
    print(f"First 30 chars: {auth_token[:30]}")
    
    # Test if it works
    import urllib.request, urllib.parse
    data = urllib.parse.urlencode({"hashedItemCode": "3fXxMs-10wXEWmClWkKRw6X9s2k="}).encode()
    req = urllib.request.Request(
        "https://talk-pilsner.kakao.com/emoticon/api/store/v3/item-code-by-hash",
        data=data,
        headers={"Authorization": auth_token}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"API test: {resp.read().decode()}")
    except Exception as e:
        print(f"API test failed: {e}")
else:
    print("\nFAILED: No auth_token found in memory")
