import datetime
import json
import logging
import os
import subprocess
import tempfile
import urllib.parse
import zipfile
from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image
from aiohttp import ClientSession
from telegram import Update, InputSticker
from telegram.constants import StickerFormat, StickerType
from telegram.ext import ContextTypes

from kakao_emoticon_bot.config import EMOTICON_ID_REGEX, SHARE_LINK_REGEX, KAKAO_AUTH_TOKEN, APP_DIR
from kakao_emoticon_bot.decrypt_kakao import DecryptKakao

logger = logging.getLogger(__name__)

HEADERS_WEB = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
HEADERS_ANDROID = {
    "User-Agent": "Android"
}

DECRYPT_EXTS = {".gif", ".webp"}

# Per-user auth token storage (file-backed)
_AUTH_FILE = os.path.join(APP_DIR, ".kakao_auth_tokens.json")

def _load_auth_tokens() -> dict:
    try:
        with open(_AUTH_FILE, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}

def _save_auth_tokens(tokens: dict) -> None:
    with open(_AUTH_FILE, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in tokens.items()}, f)

_user_auth_tokens: dict[int, str] = _load_auth_tokens()


def _clean_token(token: str) -> str:
    # Cut at first \r or \n (HTTP header boundary)
    for ch in ("\r", "\n"):
        idx = token.find(ch)
        if idx != -1:
            token = token[:idx]
    return token.strip()


def set_user_token(user_id: int, token: str) -> None:
    _user_auth_tokens[user_id] = _clean_token(token)
    _save_auth_tokens(_user_auth_tokens)


def get_auth_token(user_id: int) -> str:
    token = _user_auth_tokens.get(user_id, KAKAO_AUTH_TOKEN)
    if token:
        token = _clean_token(token)
    return token


def _validate_token(token: str) -> bool:
    """Quick check if token works against Kakao API."""
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://talk-pilsner.kakao.com/emoticon/api/store/v3/item-code-by-hash",
            data=b"hashedItemCode=test",
            headers={"Authorization": token},
        )
        resp = urllib.request.urlopen(req, timeout=5)
        return True  # 200 = valid token (even if hash is wrong, won't be 401)
    except urllib.error.HTTPError as e:
        return e.code != 401  # 401 = expired/invalid
    except Exception:
        return False


def _extract_token_from_desktop() -> Optional[str]:
    """Extract auth_token from KakaoTalk Desktop memory (sync, reusable).
    
    Collects all candidates and validates them against the API.
    """
    try:
        from PyMemoryEditor import OpenProcess
        from sticker_convert.utils.process import find_pid_by_name
        import re as _re

        kakao_pid = find_pid_by_name("kakaotalk")
        if kakao_pid is None:
            return None

        candidates = set()

        try:
            with OpenProcess(pid=int(kakao_pid)) as process:
                for address in process.search_by_value(str, 15, "authorization: "):
                    auth_token_addr = address + 15
                    auth_token_bytes = process.read_process_memory(
                        auth_token_addr, bytes, 200
                    )
                    for term_byte in (b"\r", b"\n", b"\x00"):
                        pos = auth_token_bytes.find(term_byte)
                        if pos != -1:
                            auth_token_bytes = auth_token_bytes[:pos]
                    try:
                        candidate = auth_token_bytes.decode("ascii").strip()
                    except UnicodeDecodeError:
                        continue
                    if len(candidate) > 100:
                        candidates.add(candidate)
        except PermissionError:
            pass

        try:
            from sticker_convert.utils.process import get_mem
            s = get_mem(kakao_pid, lambda msg: "", False)
            if s:
                for i in _re.finditer(b"authorization: ", s):
                    addr = i.start() + 15
                    token_bytes = s[addr:addr + 200]
                    term = token_bytes.find(b"\x00")
                    if term == -1:
                        continue
                    try:
                        candidate = token_bytes[:term].decode("ascii").strip()
                    except UnicodeDecodeError:
                        continue
                    if len(candidate) > 100:
                        candidates.add(candidate)
        except Exception:
            pass

        logger.info(f"[token] Found {len(candidates)} token candidates from desktop memory")

        # Validate each candidate
        for candidate in sorted(candidates, key=len, reverse=True):
            if _validate_token(candidate):
                logger.info(f"[token] Valid token found (len={len(candidate)})")
                return candidate
            else:
                logger.info(f"[token] Invalid/expired token (len={len(candidate)})")

    except ImportError:
        pass
    except Exception:
        pass
    return None


# ─── Kakao API helpers ─────────────────────────────────────────

async def resolve_pack_title(session: ClientSession, url: str) -> Optional[str]:
    """Resolve a share link (emoticon.kakao.com) to a pack title (e.kakao.com/t/xxx)."""
    async with session.get(url, allow_redirects=True) as resp:
        final_url = str(resp.url)
        if "e.kakao.com/t/" in final_url:
            return final_url.split("/t/")[-1].split("?")[0]
    return None


async def get_pack_meta(session: ClientSession, pack_title: str) -> Optional[dict]:
    url = f"https://e.kakao.com/api/v1/items/t/{pack_title}"
    async with session.get(url) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
        return data.get("result")


async def get_item_code_from_hash(
    session: ClientSession, hashed_code: str, auth_token: str
) -> Optional[str]:
    headers = {"Authorization": auth_token}
    data = {"hashedItemCode": hashed_code}
    logger.info(f"[hash->code] hashed={hashed_code}, token_len={len(auth_token)}")
    async with session.post(
        "https://talk-pilsner.kakao.com/emoticon/api/store/v3/item-code-by-hash",
        headers=headers, data=data
    ) as resp:
        logger.info(f"[hash->code] status={resp.status}")
        if resp.status != 200:
            body = await resp.text()
            logger.warning(f"[hash->code] failed: {resp.status} {body[:200]}")
            return None
        result = await resp.json()
        logger.info(f"[hash->code] result={result}")
        return result.get("itemCode")


async def get_item_code_from_search(
    session: ClientSession, pack_title: str, title_ko: str, auth_token: str
) -> Optional[str]:
    headers = {"Authorization": auth_token}
    data = {"query": title_ko}
    logger.info(f"[search->code] query={title_ko}, token_len={len(auth_token)}")
    async with session.post(
        "https://talk-pilsner.kakao.com/emoticon/item_store/instant_search",
        headers=headers, data=data
    ) as resp:
        logger.info(f"[search->code] status={resp.status}")
        if resp.status != 200:
            body = await resp.text()
            logger.warning(f"[search->code] failed: {resp.status} {body[:200]}")
            return None
        result = await resp.json()
        logger.info(f"[search->code] found {len(result.get('emoticons', []))} emoticons")

    for emoticon in result.get("emoticons", []):
        item_code = emoticon.get("item_code")
        if item_code:
            # Verify by checking share link
            share_info = emoticon.get("itemMetaInfo", {}).get("shareData", {})
            share_link = share_info.get("linkUrl", "")
            if share_link:
                resolved = await resolve_pack_title(
                    session, share_link
                )
                if resolved == pack_title:
                    return str(item_code)
            else:
                return str(item_code)
    return None


async def download_animated_pack(
    session: ClientSession, item_code: str, sticker_count: int = 32
) -> Optional[List[Tuple[str, bytes]]]:
    """Download animated emoticons individually and return list of (filename, decrypted_bytes).
    
    Tries file_pack.zip first, then falls back to individual file download.
    URL pattern: https://item.kakaocdn.net/dw/{item_code}.{type}_{num:03d}.{ext}
    """
    # Method 1: Try file_pack.zip
    pack_url = f"http://item.kakaocdn.net/dw/{item_code}.file_pack.zip"
    async with session.get(pack_url, headers=HEADERS_ANDROID) as resp:
        if resp.status == 200:
            zip_data = await resp.read()
            files = []
            with zipfile.ZipFile(BytesIO(zip_data), "r") as zf:
                for filepath in sorted(zf.namelist()):
                    _, ext = os.path.splitext(filepath)
                    raw = zf.read(filepath)
                    if ext.lower() in DECRYPT_EXTS:
                        raw = DecryptKakao.xor_data(raw)
                    files.append((os.path.basename(filepath), raw))
            if files:
                return files

    # Method 2: Detect format by trying first file with different type/ext combos
    play_type = ""
    play_ext = ""
    for pt in ["emot", "emoji", ""]:
        for ext in [".webp", ".gif", ".png"]:
            prefix = f"{pt}_" if pt else ""
            url = f"https://item.kakaocdn.net/dw/{item_code}.{prefix}001{ext}"
            async with session.get(url, headers=HEADERS_ANDROID) as resp:
                if resp.status == 200:
                    play_type = pt
                    play_ext = ext
                    break
        if play_ext:
            break

    if not play_ext:
        logger.warning(f"Could not detect format for item_code {item_code}")
        return None

    # Method 3: Download individual files
    files = []
    for num in range(1, sticker_count + 1):
        prefix = f"{play_type}_" if play_type else ""
        url = f"https://item.kakaocdn.net/dw/{item_code}.{prefix}{num:03d}{play_ext}"
        async with session.get(url, headers=HEADERS_ANDROID) as resp:
            if resp.status != 200:
                break
            raw = await resp.read()
            filename = f"{prefix}{num:03d}{play_ext}"
            if play_ext.lower() in DECRYPT_EXTS:
                raw = DecryptKakao.xor_data(raw)
            files.append((filename, raw))

    return files if files else None


async def try_get_item_code(
    session: ClientSession, pack_meta: dict, pack_title: str, auth_token: str,
    user_id: int = 0,
) -> Tuple[Optional[str], str]:
    """Try multiple methods to get item_code for animated download.
    
    Returns (item_code, auth_token) — auth_token may be refreshed if expired.
    """
    hashed = pack_meta.get("hashedItemCode", "")
    logger.info(f"[try_get_item_code] pack_title={pack_title}, hashed='{hashed}', auth_token_len={len(auth_token) if auth_token else 0}")

    for attempt in range(2):  # 0 = original token, 1 = refreshed token
        # Method 1: hash -> item_code
        if hashed:
            item_code = await get_item_code_from_hash(session, hashed, auth_token)
            if item_code:
                return item_code, auth_token

        # Method 2: search by Korean title
        title_ko = pack_meta.get("title", "")
        if title_ko:
            item_code = await get_item_code_from_search(
                session, pack_title, title_ko, auth_token
            )
            if item_code:
                return item_code, auth_token

        # If first attempt failed, try refreshing token from desktop
        if attempt == 0:
            import asyncio
            loop = asyncio.get_event_loop()
            logger.info("[try_get_item_code] Token may be expired, trying desktop refresh...")
            new_token = await loop.run_in_executor(None, _extract_token_from_desktop)
            if new_token and new_token != auth_token:
                logger.info(f"[try_get_item_code] Got fresh token (len={len(new_token)})")
                auth_token = new_token
                if user_id:
                    set_user_token(user_id, auth_token)
            else:
                logger.info("[try_get_item_code] No new token from desktop, giving up")
                break

    return None, auth_token


# ─── Sticker creation helpers ──────────────────────────────────

def is_animated_image(data: bytes) -> bool:
    """Check if image data is animated WebP or GIF."""
    if data[:4] == b"RIFF" and b"WEBP" in data[:12]:
        return b"ANIM" in data[:200]
    if data[:3] == b"GIF":
        return True
    return False


async def make_stickers_static(
    session: ClientSession, thumbnail_urls: List[str]
) -> List[InputSticker]:
    stickers = []
    for url in thumbnail_urls:
        async with session.get(url) as img_resp:
            img_bytes = BytesIO()
            img = Image.open(BytesIO(await img_resp.read()))
            img = img.resize((512, 512))
            img.save(img_bytes, "png")
            stickers.append(
                InputSticker(
                    sticker=img_bytes.getvalue(),
                    emoji_list=["\U0001f600"],
                    format=StickerFormat.STATIC,
                )
            )
    return stickers


def convert_to_webm(data: bytes, filename: str, size: int = 512) -> Optional[bytes]:
    """Convert animated WebP/GIF to WebM video sticker using ffmpeg.
    
    Telegram video sticker requirements:
    - WebM VP9 codec, max 256KB
    - 512x512, max 3 seconds, 30fps
    - No audio
    
    Uses Pillow to extract frames first (ffmpeg can't decode Kakao animated WebP directly),
    then ffmpeg to encode frames into VP9 WebM.
    """
    try:
        img = Image.open(BytesIO(data))
        n_frames = getattr(img, "n_frames", 1)
        if n_frames <= 1:
            return None
    except Exception as e:
        logger.warning(f"Failed to open {filename} with Pillow: {e}")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract frames with Pillow, limiting total duration to 2.9s (safe margin for 3s limit)
        MAX_DURATION_MS = 2900
        durations = []
        total_ms = 0
        frame_count = 0
        for i in range(n_frames):
            img.seek(i)
            dur = img.info.get("duration", 50)
            if total_ms + dur > MAX_DURATION_MS and frame_count > 0:
                break
            frame = img.copy().convert("RGBA").resize((size, size))
            frame.save(os.path.join(tmpdir, f"frame_{i:04d}.png"), "PNG")
            durations.append(dur)
            total_ms += dur
            frame_count += 1

        avg_dur = sum(durations) / len(durations) if durations else 100
        fps = min(1000.0 / avg_dur, 30) if avg_dur > 0 else 10
        logger.info(f"{filename}: {frame_count}/{n_frames} frames, {total_ms}ms, {fps:.1f}fps")

        output_path = os.path.join(tmpdir, "output.webm")

        for crf in [30, 40, 50, 63]:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%04d.png"),
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",
                "-crf", str(crf),
                "-b:v", "0",
                "-t", "3",
                "-an",
                "-auto-alt-ref", "0",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                logger.warning(f"ffmpeg failed for {filename} crf={crf}: {result.stderr[-200:]}")
                continue

            with open(output_path, "rb") as f:
                webm_data = f.read()

            if len(webm_data) <= 256 * 1024:
                logger.info(f"{filename}: WebM {len(webm_data)/1024:.1f}KB (crf={crf}, {n_frames} frames, {fps:.1f}fps)")
                return webm_data
            logger.info(f"{filename} crf={crf}: {len(webm_data)} bytes, too big, retrying...")

    return None


def make_stickers_animated(files: List[Tuple[str, bytes]]) -> List[InputSticker]:
    stickers = []
    for filename, data in files:
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        # Skip sound files
        if ext in (".mp3", ".m4a", ".ogg"):
            continue

        is_anim = is_animated_image(data)

        if is_anim and ext in (".webp", ".gif"):
            # Convert to WebM video sticker via ffmpeg
            webm_data = convert_to_webm(data, filename)
            if webm_data:
                stickers.append(
                    InputSticker(
                        sticker=webm_data,
                        emoji_list=["\U0001f600"],
                        format=StickerFormat.VIDEO,
                    )
                )
                continue
            else:
                logger.warning(f"WebM conversion failed for {filename}, falling back to static")

        # Fallback: static PNG
        try:
            img = Image.open(BytesIO(data))
            out = BytesIO()
            img = img.resize((512, 512))
            img.save(out, "png")
            stickers.append(
                InputSticker(
                    sticker=out.getvalue(),
                    emoji_list=["\U0001f600"],
                    format=StickerFormat.STATIC,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to process {filename}: {e}")

    return stickers


def make_emoji_stickers(files: List[Tuple[str, bytes]]) -> List[InputSticker]:
    """Convert downloaded files to 100x100 custom emoji InputStickers."""
    stickers = []
    for filename, data in files:
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext in (".mp3", ".m4a", ".ogg"):
            continue

        is_anim = is_animated_image(data)

        if is_anim and ext in (".webp", ".gif"):
            webm_data = convert_to_webm(data, filename, size=100)
            if webm_data:
                stickers.append(
                    InputSticker(
                        sticker=webm_data,
                        emoji_list=["\U0001f600"],
                        format=StickerFormat.VIDEO,
                    )
                )
                continue
            else:
                logger.warning(f"WebM conversion failed for emoji {filename}, falling back to static")

        # Static PNG at 100x100
        try:
            img = Image.open(BytesIO(data))
            out = BytesIO()
            img = img.convert("RGBA").resize((100, 100))
            img.save(out, "png")
            stickers.append(
                InputSticker(
                    sticker=out.getvalue(),
                    emoji_list=["\U0001f600"],
                    format=StickerFormat.STATIC,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to process emoji {filename}: {e}")

    return stickers


async def make_emoji_stickers_static(
    session: ClientSession, thumbnail_urls: List[str]
) -> List[InputSticker]:
    """Download thumbnails and convert to 100x100 custom emoji stickers."""
    stickers = []
    for url in thumbnail_urls:
        async with session.get(url) as img_resp:
            img_bytes = BytesIO()
            img = Image.open(BytesIO(await img_resp.read()))
            img = img.convert("RGBA").resize((100, 100))
            img.save(img_bytes, "png")
            stickers.append(
                InputSticker(
                    sticker=img_bytes.getvalue(),
                    emoji_list=["\U0001f600"],
                    format=StickerFormat.STATIC,
                )
            )
    return stickers


async def upload_emoji_set(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int, user_id: int,
    title: str, stickers: List[InputSticker],
) -> None:
    """Upload a custom emoji set (100x100) to Telegram."""
    cur_time = str(
        datetime.datetime.now(datetime.timezone.utc).timestamp()
    ).replace(".", "")
    sticker_name = f"e{cur_time}_by_{context.bot.name[1:]}"
    total = len(stickers)

    doing_message = await context.bot.send_message(
        chat_id=chat_id,
        text=f"커스텀 이모지 업로드 중... (0/{total})",
    )

    await context.bot.create_new_sticker_set(
        user_id=user_id,
        name=sticker_name,
        title=title,
        stickers=[stickers[0]],
        sticker_type=StickerType.CUSTOM_EMOJI,
    )

    await doing_message.edit_text(text=f"업로드 중... (1/{total})")

    for index, sticker in enumerate(stickers[1:], 2):
        await context.bot.add_sticker_to_set(
            user_id=user_id,
            name=sticker_name,
            sticker=sticker,
        )
        if index % 3 == 0 or index == total:
            await doing_message.edit_text(text=f"업로드 중... ({index}/{total})")

    await doing_message.edit_text(
        text=f"'{title}' 커스텀 이모지 생성이 완료되었습니다!",
    )
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"https://t.me/addstickers/{sticker_name}",
    )
    logger.info(f"Custom emoji set created: {sticker_name} for user {user_id}")


async def upload_sticker_set(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int, user_id: int,
    title: str, stickers: List[InputSticker],
) -> None:
    cur_time = str(
        datetime.datetime.now(datetime.timezone.utc).timestamp()
    ).replace(".", "")
    sticker_name = f"t{cur_time}_by_{context.bot.name[1:]}"
    total = len(stickers)

    doing_message = await context.bot.send_message(
        chat_id=chat_id,
        text=f"텔레그램 서버로 업로드 중... (0/{total})",
    )

    await context.bot.create_new_sticker_set(
        user_id=user_id,
        name=sticker_name,
        title=title,
        stickers=[stickers[0]],
    )

    await doing_message.edit_text(text=f"업로드 중... (1/{total})")

    for index, sticker in enumerate(stickers[1:], 2):
        await context.bot.add_sticker_to_set(
            user_id=user_id,
            name=sticker_name,
            sticker=sticker,
        )
        if index % 3 == 0 or index == total:
            await doing_message.edit_text(text=f"업로드 중... ({index}/{total})")

    await doing_message.edit_text(
        text=f"'{title}' 스티커 생성이 완료되었습니다!",
    )
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"https://t.me/addstickers/{sticker_name}",
    )
    logger.info(f"Sticker set created: {sticker_name} for user {user_id}")


# ─── Core download helper (reused by /create and /merge) ─────

async def download_stickers_from_url(
    input_url: str, auth_token: str, chat_id: int, context: ContextTypes.DEFAULT_TYPE
) -> Optional[List[InputSticker]]:
    """Download stickers from a single Kakao URL. Returns list of InputSticker or None."""
    pack_title = None
    hashed_item_code = None

    if EMOTICON_ID_REGEX.match(input_url):
        pack_title = input_url.split("/t/")[-1].split("?")[0]
    elif SHARE_LINK_REGEX.match(input_url):
        async with ClientSession(headers={"User-Agent": "Chrome"}) as session:
            pack_title = await resolve_pack_title(session, input_url)
            if not pack_title:
                hashed_item_code = urlparse(input_url).path.split("/")[-1]
    elif input_url.isnumeric():
        async with ClientSession(headers=HEADERS_ANDROID) as session:
            files = await download_animated_pack(session, input_url)
            if files:
                return make_stickers_animated(files)
        return None
    else:
        return None

    async with ClientSession(headers=HEADERS_WEB) as session:
        pack_meta = await get_pack_meta(session, pack_title) if pack_title else None
        if not pack_meta:
            return None

        title = pack_meta.get("title", pack_title)
        thumbnail_urls = pack_meta.get("thumbnailUrls", [])

        # Try animated download if auth_token is available
        if auth_token:
            item_code, auth_token = await try_get_item_code(
                session, pack_meta, pack_title, auth_token, user_id=0
            )
            if item_code:
                files = await download_animated_pack(session, item_code, len(thumbnail_urls))
                if files:
                    stickers = make_stickers_animated(files)
                    if stickers:
                        return stickers

        # Static fallback
        return await make_stickers_static(session, thumbnail_urls)


# ─── Command handlers ─────────────────────────────────────────

async def create_emoticon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.effective_user:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if not context.args:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "사용법:\n"
                "/create https://e.kakao.com/t/이모티콘URL\n"
                "/create https://emoticon.kakao.com/items/해시코드\n\n"
                "/search 키워드 로 이모티콘을 검색할 수 있습니다.\n"
                "/setauth 토큰 으로 카카오 인증 토큰을 설정하면 움직이는 이모티콘을 받을 수 있습니다."
            ),
        )
        return

    input_url = context.args[0]
    auth_token = get_auth_token(user_id)
    print(f"[DEBUG] user_id={user_id}, auth_token={'SET(len=' + str(len(auth_token)) + ')' if auth_token else 'NONE'}")

    # Determine pack_title from URL
    pack_title = None
    hashed_item_code = None

    if EMOTICON_ID_REGEX.match(input_url):
        pack_title = input_url.split("/t/")[-1].split("?")[0]
    elif SHARE_LINK_REGEX.match(input_url):
        await context.bot.send_message(chat_id=chat_id, text="공유 링크를 분석하는 중...")
        async with ClientSession(headers={"User-Agent": "Chrome"}) as session:
            pack_title = await resolve_pack_title(session, input_url)
            if not pack_title:
                hashed_item_code = urlparse(input_url).path.split("/")[-1]
    elif input_url.isnumeric():
        # Direct item_code
        if not auth_token:
            await context.bot.send_message(
                chat_id=chat_id,
                text="숫자 item_code를 사용하려면 /setauth 로 인증 토큰을 먼저 설정하세요.",
            )
            return
        await context.bot.send_message(chat_id=chat_id, text="움직이는 이모티콘을 다운로드합니다...")
        async with ClientSession(headers=HEADERS_ANDROID) as session:
            files = await download_animated_pack(session, input_url)
            if not files:
                await context.bot.send_message(chat_id=chat_id, text="다운로드 실패: 유효하지 않은 item_code입니다.")
                return
            stickers = make_stickers_animated(files)
            if not stickers:
                await context.bot.send_message(chat_id=chat_id, text="스티커로 변환할 이미지가 없습니다.")
                return
            await upload_sticker_set(context, chat_id, user_id, f"kakao_{input_url}", stickers)
        return
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text="유효한 URL이 아닙니다.\n예시: https://e.kakao.com/t/charming-apeach",
        )
        return

    await context.bot.send_message(chat_id=chat_id, text="이모티콘 정보를 불러오는 중입니다...")

    try:
        async with ClientSession(headers=HEADERS_WEB) as session:
            pack_meta = await get_pack_meta(session, pack_title) if pack_title else None

            if not pack_meta:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="이모티콘 정보를 불러올 수 없습니다. URL을 확인해주세요.",
                )
                return

            title = pack_meta.get("title", pack_title)
            thumbnail_urls = pack_meta.get("thumbnailUrls", [])
            hashed = pack_meta.get("hashedItemCode", hashed_item_code or "")

            # Try animated download if auth_token is available
            if auth_token:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"'{title}' - 움직이는 이모티콘 다운로드를 시도합니다...",
                )

                item_code, auth_token = await try_get_item_code(
                    session, pack_meta, pack_title, auth_token, user_id=user_id
                )

                if item_code:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"item_code: {item_code} - 애니메이션 팩을 다운로드 중...",
                    )
                    files = await download_animated_pack(session, item_code, len(thumbnail_urls))
                    if files:
                        stickers = make_stickers_animated(files)
                        if stickers:
                            await upload_sticker_set(
                                context, chat_id, user_id, title, stickers
                            )
                            return
                        else:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text="애니메이션 변환 실패. 정적 이미지로 전환합니다...",
                            )
                    else:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text="애니메이션 팩 다운로드 실패. 정적 이미지로 전환합니다...",
                        )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="item_code를 찾을 수 없습니다. 정적 이미지로 전환합니다...",
                    )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"'{title}' 이모티콘을 정적 이미지로 다운로드합니다. ({len(thumbnail_urls)}개)\n"
                        f"💡 /setauth 토큰 으로 카카오 인증 토큰을 설정하면 움직이는 이모티콘을 받을 수 있습니다."
                    ),
                )

            # Static fallback
            stickers = await make_stickers_static(session, thumbnail_urls)
            if not stickers:
                await context.bot.send_message(chat_id=chat_id, text="스티커로 변환할 이미지가 없습니다.")
                return
            await upload_sticker_set(context, chat_id, user_id, title, stickers)

    except Exception as e:
        logger.error(f"Error creating sticker set: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"스티커 생성 중 오류가 발생했습니다.\n{e}",
        )


async def search_emoticon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return

    chat_id = update.effective_chat.id

    if not context.args:
        await context.bot.send_message(
            chat_id=chat_id,
            text="사용법: /search 키워드\n예시: /search 어피치",
        )
        return

    query = " ".join(context.args)
    encoded = urllib.parse.quote(query, safe="")
    api_url = f"https://e.kakao.com/api/v1/search?query={encoded}&page=0&size=10"

    try:
        async with ClientSession(headers=HEADERS_WEB) as session:
            async with session.get(api_url) as resp:
                if resp.status != 200:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"검색 실패 (HTTP {resp.status})",
                    )
                    return
                data = await resp.json()

        items = data.get("result", {}).get("content", [])
        if not items:
            await context.bot.send_message(
                chat_id=chat_id,
                text="검색 결과가 없습니다.",
            )
            return

        lines = [f"\U0001f50d '{query}' 검색 결과 ({len(items)}개):\n"]
        for i, item in enumerate(items, 1):
            title = item.get("title", "?")
            artist = item.get("artist", "?")
            title_url = item.get("titleUrl", "")
            url = f"https://e.kakao.com/t/{title_url}"
            lines.append(f"{i}. {title} ({artist})\n   /create {url}")

        await context.bot.send_message(
            chat_id=chat_id,
            text="\n".join(lines),
        )

    except Exception as e:
        logger.error(f"Error searching emoticons: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"검색 중 오류가 발생했습니다.\n{e}",
        )


async def merge_emoticons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Merge multiple Kakao emoticon packs into one Telegram sticker set."""
    if not update.effective_chat or not update.effective_user:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if not context.args or len(context.args) < 2:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "사용법: /merge 스티커이름 URL1 URL2 URL3 ...\n\n"
                "예시:\n"
                "/merge 빵빵이모음 https://e.kakao.com/t/bbangbbangs-daily-life-3 https://e.kakao.com/t/bbangbbang-and-okji\n\n"
                "첫 번째 인자는 스티커 세트 이름, 나머지는 합칠 이모티콘 URL입니다.\n"
                "Telegram 스티커 세트는 최대 120개까지 가능합니다."
            ),
        )
        return

    set_title = context.args[0]
    urls = context.args[1:]
    auth_token = get_auth_token(user_id)

    all_stickers: List[InputSticker] = []
    total_urls = len(urls)

    await context.bot.send_message(
        chat_id=chat_id,
        text=f"'{set_title}' - {total_urls}개 팩을 합치는 중...",
    )

    try:
        for idx, url in enumerate(urls, 1):
            status_msg = await context.bot.send_message(
                chat_id=chat_id,
                text=f"[{idx}/{total_urls}] 다운로드 중: {url.split('/')[-1]}",
            )

            stickers = await download_stickers_from_url(url, auth_token, chat_id, context)
            if stickers:
                all_stickers.extend(stickers)
                await status_msg.edit_text(
                    text=f"[{idx}/{total_urls}] {url.split('/')[-1]}: {len(stickers)}개 스티커 추가 (누적 {len(all_stickers)}개)",
                )
            else:
                await status_msg.edit_text(
                    text=f"[{idx}/{total_urls}] {url.split('/')[-1]}: 다운로드 실패, 건너뜀",
                )

        if not all_stickers:
            await context.bot.send_message(
                chat_id=chat_id,
                text="합칠 스티커가 없습니다. URL을 확인해주세요.",
            )
            return

        # Telegram limit: max 120 stickers per set
        if len(all_stickers) > 120:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"총 {len(all_stickers)}개 스티커 중 120개만 사용합니다. (Telegram 제한)",
            )
            all_stickers = all_stickers[:120]

        # Check: all stickers must be same format (static or video)
        # Telegram doesn't allow mixing static and video in one set
        has_video = any(s.format == StickerFormat.VIDEO for s in all_stickers)
        has_static = any(s.format == StickerFormat.STATIC for s in all_stickers)
        if has_video and has_static:
            await context.bot.send_message(
                chat_id=chat_id,
                text="움직이는 스티커와 정적 스티커가 섞여 있어 정적 스티커만 사용합니다.\n"
                     "(Telegram은 한 세트에 한 종류만 허용합니다)",
            )
            all_stickers = [s for s in all_stickers if s.format == StickerFormat.STATIC]

        await upload_sticker_set(context, chat_id, user_id, set_title, all_stickers)

    except Exception as e:
        logger.error(f"Error merging sticker sets: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"스티커 합치기 중 오류가 발생했습니다.\n{e}",
        )


async def download_emoji_from_url(
    url: str, auth_token: str, chat_id: int,
    context: ContextTypes.DEFAULT_TYPE, user_id: int = 0,
) -> Optional[List[InputSticker]]:
    """Download stickers from a Kakao URL and convert to 100x100 custom emoji."""
    pack_title = None
    hashed_item_code = None

    if EMOTICON_ID_REGEX.match(url):
        pack_title = url.split("/t/")[-1].split("?")[0]
    elif SHARE_LINK_REGEX.match(url):
        async with ClientSession(headers={"User-Agent": "Chrome"}) as session:
            resolved = await resolve_pack_title(session, url)
            if resolved:
                pack_title = resolved
            else:
                hashed_item_code = urlparse(url).path.split("/")[-1]

    async with ClientSession(headers=HEADERS_WEB) as session:
        pack_meta = await get_pack_meta(session, pack_title) if pack_title else None
        if not pack_meta:
            return None

        title = pack_meta.get("title", pack_title)
        thumbnail_urls = pack_meta.get("thumbnailUrls", [])

        # Try animated download if auth_token is available
        if auth_token:
            item_code, auth_token = await try_get_item_code(
                session, pack_meta, pack_title, auth_token, user_id=user_id
            )
            logger.info(f"[emoji] item_code={item_code} for {pack_title}")
            if item_code:
                files = await download_animated_pack(session, item_code, len(thumbnail_urls))
                if files:
                    logger.info(f"[emoji] Downloaded {len(files)} files, first={files[0][0]}")
                    stickers = make_emoji_stickers(files)
                    logger.info(f"[emoji] Made {len(stickers)} emoji stickers, formats={[s.format for s in stickers[:3]]}")
                    if stickers:
                        return stickers
                else:
                    logger.warning(f"[emoji] download_animated_pack returned no files for {item_code}")
            else:
                logger.warning(f"[emoji] item_code not found, falling back to static thumbnails")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="⚠️ auth_token이 만료되어 움직이는 이모지를 다운로드할 수 없습니다.\n"
                         "카카오톡 데스크톱을 재시작한 후 /getauth 를 실행하거나,\n"
                         "정적 이미지로 대신 생성합니다.",
                )
        else:
            logger.warning(f"[emoji] No auth_token, using static thumbnails")

        # Static fallback: use thumbnails at 100x100
        return await make_emoji_stickers_static(session, thumbnail_urls)


async def create_emoji(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Create 100x100 Telegram custom emoji from any Kakao emoticon pack."""
    if not update.effective_chat or not update.effective_user:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if not context.args:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "사용법: /emoji https://e.kakao.com/t/이모티콘URL\n\n"
                "카카오 이모티콘을 텔레그램 커스텀 이모지(100x100)로 변환합니다.\n"
                "움직이는 이모티콘은 움직이는 커스텀 이모지로 변환됩니다.\n"
                "/search 키워드 로 이모티콘을 검색할 수 있습니다."
            ),
        )
        return

    input_url = context.args[0]
    auth_token = get_auth_token(user_id)

    if not EMOTICON_ID_REGEX.match(input_url) and not SHARE_LINK_REGEX.match(input_url):
        await context.bot.send_message(
            chat_id=chat_id,
            text="유효한 URL이 아닙니다.\n예시: /emoji https://e.kakao.com/t/example-emoticon",
        )
        return

    await context.bot.send_message(chat_id=chat_id, text="이모티콘 정보를 불러오는 중입니다...")

    try:
        stickers = await download_emoji_from_url(
            input_url, auth_token, chat_id, context, user_id=user_id
        )
        if not stickers:
            await context.bot.send_message(
                chat_id=chat_id,
                text="이모지로 변환할 이미지가 없습니다. URL을 확인해주세요.",
            )
            return

        # Determine title
        pack_title = input_url.split("/t/")[-1].split("?")[0] if "/t/" in input_url else "emoji"
        async with ClientSession(headers=HEADERS_WEB) as session:
            pack_meta = await get_pack_meta(session, pack_title)
        title = pack_meta.get("title", pack_title) if pack_meta else pack_title

        has_video = any(s.format == StickerFormat.VIDEO for s in stickers)
        has_static = any(s.format == StickerFormat.STATIC for s in stickers)
        if has_video and has_static:
            # Prefer video (animated) stickers
            stickers = [s for s in stickers if s.format == StickerFormat.VIDEO]

        await upload_emoji_set(context, chat_id, user_id, title, stickers)

    except Exception as e:
        logger.error(f"Error creating custom emoji set: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"커스텀 이모지 생성 중 오류가 발생했습니다.\n{e}",
        )


async def set_auth_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.effective_user:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if not context.args:
        has_token = bool(get_auth_token(user_id))
        status = "설정됨 ✅" if has_token else "미설정 ❌"
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"현재 인증 토큰 상태: {status}\n\n"
                "사용법: /setauth 카카오인증토큰\n\n"
                "카카오 인증 토큰을 설정하면 움직이는 이모티콘을 다운로드할 수 있습니다.\n"
                "토큰은 카카오톡 앱의 HTTP 요청에서 Authorization 헤더 값을 추출하면 됩니다."
            ),
        )
        return

    token = context.args[0]
    set_user_token(user_id, token)

    await context.bot.send_message(
        chat_id=chat_id,
        text="카카오 인증 토큰이 설정되었습니다. ✅\n이제 /create 명령으로 움직이는 이모티콘을 다운로드할 수 있습니다.",
    )


async def get_auth_from_desktop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Extract auth_token from KakaoTalk Desktop running on this PC."""
    if not update.effective_chat or not update.effective_user:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    await context.bot.send_message(
        chat_id=chat_id,
        text="PC에서 카카오톡 데스크톱의 auth_token을 추출합니다...\n(카카오톡 데스크톱이 로그인된 상태여야 합니다)",
    )

    import asyncio
    loop = asyncio.get_event_loop()
    token = await loop.run_in_executor(None, _extract_token_from_desktop)
    error = None if token else "카카오톡 메모리에서 auth_token을 찾을 수 없습니다.\n카카오톡이 실행 중이고 로그인되어 있는지 확인하세요."

    if token:
        set_user_token(user_id, token)
        # Show only first/last 10 chars for security
        masked = token[:10] + "..." + token[-10:]
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"auth_token 추출 성공! ✅\n"
                f"토큰: {masked}\n\n"
                f"이제 /create 명령으로 움직이는 이모티콘을 다운로드할 수 있습니다."
            ),
        )
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ {error}",
        )
