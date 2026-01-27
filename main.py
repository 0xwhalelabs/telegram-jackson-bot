import asyncio
import base64
import io
import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from firebase_admin import credentials, firestore
import firebase_admin
from telegram import ChatPermissions, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


load_dotenv()


_USER_LOCKS: Dict[Tuple[int, int], asyncio.Lock] = {}


_YACHA_DUELS: Dict[int, Dict[str, Any]] = {}


_YACHA_CHAT_LOCKS: Dict[int, asyncio.Lock] = {}


_CHAT_LOCKS: Dict[int, asyncio.Lock] = {}


def get_chat_lock(chat_id: int) -> asyncio.Lock:
    key = int(chat_id)
    lock = _CHAT_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _CHAT_LOCKS[key] = lock
    return lock


def get_yacha_chat_lock(chat_id: int) -> asyncio.Lock:
    key = int(chat_id)
    lock = _YACHA_CHAT_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _YACHA_CHAT_LOCKS[key] = lock
    return lock


def get_user_lock(chat_id: int, user_id: int) -> asyncio.Lock:
    key = (int(chat_id), int(user_id))
    lock = _USER_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _USER_LOCKS[key] = lock
    return lock


async def acquire_two_user_locks(chat_id: int, user_a: int, user_b: int) -> Tuple[asyncio.Lock, asyncio.Lock]:
    a = int(user_a)
    b = int(user_b)
    first, second = (a, b) if a <= b else (b, a)
    lock1 = get_user_lock(chat_id, first)
    lock2 = get_user_lock(chat_id, second)
    await lock1.acquire()
    try:
        await lock2.acquire()
    except Exception:
        lock1.release()
        raise
    return lock1, lock2


def release_two_user_locks(lock1: asyncio.Lock, lock2: asyncio.Lock) -> None:
    try:
        lock2.release()
    finally:
        lock1.release()


KST_TZ = "Asia/Seoul"

KEYWORD_PATTERN = re.compile(r"(?i)(\bbased\b|베이스드)")
URL_PATTERN = re.compile(
    r"(?i)(https?://|t\.me/|www\.|\b[\w-]+\.(com|io|net|me|kr|org|gg|xyz|app|dev)\b)"
)


@dataclass(frozen=True)
class ExpResult:
    gained_exp: int
    reason: str


def now_kst() -> datetime:
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo(KST_TZ))


def is_fever_time(dt: datetime) -> bool:
    start = dt.replace(hour=19, minute=0, second=0, microsecond=0)
    end = dt.replace(hour=23, minute=0, second=0, microsecond=0)
    return start <= dt < end


def is_link_block_time(dt: datetime) -> bool:
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = dt.replace(hour=9, minute=0, second=0, microsecond=0)
    return start <= dt < end


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s]+", " ", text).strip()
    text = re.sub(r"[^0-9a-z가-힣 ]+", "", text)
    text = re.sub(r"[\s]+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def compute_level(total_exp: int) -> Tuple[int, int, int]:
    level = 1
    need = 100
    remaining_exp = total_exp
    while remaining_exp >= need:
        remaining_exp -= need
        level += 1
        need *= 2
    return level, remaining_exp, need


def calculate_exp(message_text: str, dt: datetime) -> ExpResult:
    if len(message_text) < 5:
        return ExpResult(0, "short")

    has_keyword = KEYWORD_PATTERN.search(message_text) is not None
    base = 10 if has_keyword else 5

    if is_fever_time(dt):
        base = int(round(base * 1.5))

    return ExpResult(base, "keyword" if has_keyword else "base")


def get_firebase_client() -> firestore.Client:
    if not firebase_admin._apps:
        service_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
        service_b64 = os.getenv("FIREBASE_CREDENTIALS_BASE64", "").strip()

        if service_json:
            cred_info = json.loads(service_json)
            cred = credentials.Certificate(cred_info)
        elif service_b64:
            decoded = base64.b64decode(service_b64).decode("utf-8")
            cred_info = json.loads(decoded)
            cred = credentials.Certificate(cred_info)
        else:
            raise RuntimeError(
                "Missing FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_CREDENTIALS_BASE64"
            )

        firebase_admin.initialize_app(cred)

    return firestore.client()


def chat_ref(db: firestore.Client, chat_id: int):
    return db.collection("chats").document(str(chat_id))


def user_ref(db: firestore.Client, chat_id: int, user_id: int):
    return chat_ref(db, chat_id).collection("users").document(str(user_id))


def is_username_token(text: str) -> bool:
    t = (text or "").strip()
    if not t.startswith("@"):
        return False
    if len(t) < 2:
        return False
    if " " in t:
        return False
    return True


def parse_username_token(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("@"): 
        t = t[1:]
    return t


def yacha_duel_key(chat_id: int) -> int:
    return int(chat_id)


def get_active_duel(chat_id: int) -> Optional[Dict[str, Any]]:
    return _YACHA_DUELS.get(yacha_duel_key(chat_id))


def set_active_duel(chat_id: int, duel: Optional[Dict[str, Any]]) -> None:
    key = yacha_duel_key(chat_id)
    if duel is None:
        _YACHA_DUELS.pop(key, None)
    else:
        _YACHA_DUELS[key] = duel


def rps_result(a_choice: str, b_choice: str) -> int:
    beats = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    if a_choice == b_choice:
        return 0
    if beats.get(a_choice) == b_choice:
        return 1
    return -1


PALS_EGG_PRICE_EXP = 100
PALS_FEED_PRICE_EXP = 5

PALS_TYPES: List[str] = [
    "블루",
    "그린",
    "퍼플",
    "핑크",
    "레드",
]

PALS_TYPE_SLUG: Dict[int, str] = {
    1: "blue",
    2: "green",
    3: "purple",
    4: "pink",
    5: "red",
}

PALS_STAGE_LABEL: Dict[str, str] = {
    "baby": "유아기",
    "teen": "성장기",
    "adult": "완전체",
    "ultimate": "궁극체",
}

PALS_STAGE_FOLDER: Dict[str, str] = {
    "baby": "baby",
    "teen": "child",
    "adult": "drake",
    "ultimate": "adult",
}

PALS_EVOLVE_AT: Dict[str, int] = {
    "baby": 10_000,
    "teen": 20_000,
    "adult": 50_000,
}

PALS_PAYOUT_EXP: Dict[str, int] = {
    "baby": 0,
    "teen": 100,
    "adult": 200,
    "ultimate": 1000,
}


def get_pals_asset_base_url() -> str:
    v = os.getenv("PALS_ASSET_BASE_URL", "")
    v = (v or "").strip().strip('"').strip("'")
    if v.startswith("PALS_ASSET_BASE_URL="):
        v = v.split("=", 1)[1].strip()
    return v.rstrip("/")


def pals_egg_gif_url() -> str:
    base = get_pals_asset_base_url()
    if not base:
        return ""
    return f"{base}/palegg.gif"


def pals_stage_image_url(stage: str, type_id: int) -> str:
    base = get_pals_asset_base_url()
    if not base:
        return ""
    prefix = PALS_STAGE_FOLDER.get(str(stage), "baby")
    tid = max(1, min(int(type_id), 5))
    slug = PALS_TYPE_SLUG.get(tid, "blue")
    return f"{base}/{prefix}{slug}.png"


def pals_type_name(type_id: int) -> str:
    tid = max(1, min(int(type_id), 5))
    return PALS_TYPES[tid - 1]


def pals_display_title(stage: str, type_id: int) -> str:
    return f"{pals_type_name(type_id)} Pals – {PALS_STAGE_LABEL.get(str(stage), '유아기')}"


def format_timedelta_kor(seconds: int) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    if h > 0:
        return f"{h}시간 {m}분"
    return f"{m}분"


def _download_url_bytes_sync(url: str, timeout_seconds: int = 15) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return resp.read()


async def download_url_bytes(url: str, timeout_seconds: int = 15) -> Optional[bytes]:
    if not url:
        return None
    try:
        return await asyncio.to_thread(_download_url_bytes_sync, url, timeout_seconds)
    except Exception:
        return None


SWORD_MAX_LEVEL = 20
SWORD_NONE_LEVEL = -1
BASED_MALL_SWORD_LEVEL = 0
BASED_MALL_PRICE_EXP = 100


SWORD_TABLE: Dict[int, Dict[str, Any]] = {
    0: {"name": "오래된 Based 나무 검", "cost": 0, "rate": 1.0, "sell": 5},
    1: {"name": "실버 Based 검", "cost": 50, "rate": 0.70, "sell": 80},
    2: {"name": "실버+ 검", "cost": 80, "rate": 0.60, "sell": 180},
    3: {"name": "골드 Based 검", "cost": 120, "rate": 0.50, "sell": 350},
    4: {"name": "골드+ 검", "cost": 180, "rate": 0.40, "sell": 650},
    5: {"name": "플래티넘 Based 검", "cost": 250, "rate": 0.30, "sell": 1100},
    6: {"name": "플래티넘+ 검", "cost": 350, "rate": 0.22, "sell": 1800},
    7: {"name": "루비 Based 검", "cost": 500, "rate": 0.17, "sell": 3000},
    8: {"name": "루비+ 검", "cost": 700, "rate": 0.13, "sell": 5000},
    9: {"name": "사파이어 Based 검", "cost": 1000, "rate": 0.10, "sell": 8500},
    10: {"name": "사파이어+ 검", "cost": 1400, "rate": 0.08, "sell": 15000},
    11: {"name": "오닉스 Based 검", "cost": 2000, "rate": 0.06, "sell": 26000},
    12: {"name": "오닉스+ 검", "cost": 2800, "rate": 0.045, "sell": 45000},
    13: {"name": "블러드 Based 검", "cost": 3800, "rate": 0.03, "sell": 80000},
    14: {"name": "블러드+ 검", "cost": 5200, "rate": 0.02, "sell": 150000},
    15: {"name": "검은 왕의 검", "cost": 7000, "rate": 0.013, "sell": 280000},
    16: {"name": "세계절단 검", "cost": 9000, "rate": 0.009, "sell": 500000},
    17: {"name": "신의 시험 검", "cost": 12000, "rate": 0.006, "sell": 900000},
    18: {"name": "멸망의 Based 검", "cost": 16000, "rate": 0.0035, "sell": 1600000},
    19: {"name": "신화의 끝 검", "cost": 22000, "rate": 0.0015, "sell": 3000000},
    20: {"name": "⚫ 절대자 Based 검", "cost": 30000, "rate": 0.0005, "sell": None},
}


def sword_state_from_udata(udata: Dict[str, Any]) -> Tuple[int, int]:
    lvl = int(udata.get("sword_level", 0))
    if lvl < SWORD_NONE_LEVEL:
        lvl = SWORD_NONE_LEVEL
    if lvl > SWORD_MAX_LEVEL:
        lvl = SWORD_MAX_LEVEL
    tickets = int(udata.get("defense_tickets", 0))
    if tickets < 0:
        tickets = 0
    return lvl, tickets


def sword_name(level: int) -> str:
    if int(level) == SWORD_NONE_LEVEL:
        return "없음"
    return str(SWORD_TABLE.get(int(level), SWORD_TABLE[0])["name"])


def sword_sell_price(level: int) -> Optional[int]:
    if int(level) == SWORD_NONE_LEVEL:
        return None
    return SWORD_TABLE.get(int(level), SWORD_TABLE[0]).get("sell")


def sword_next_upgrade_info(level: int) -> Optional[Tuple[int, float, int, Optional[int], str]]:
    if int(level) == SWORD_NONE_LEVEL:
        return None
    nxt = int(level) + 1
    if nxt > SWORD_MAX_LEVEL:
        return None
    row = SWORD_TABLE.get(nxt)
    if not row:
        return None
    return nxt, float(row["rate"]), int(row["cost"]), row.get("sell"), str(row["name"])


def kst_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def get_allowed_chat_id() -> Optional[int]:
    raw = os.getenv("ALLOWED_CHAT_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def get_owner_user_id() -> Optional[int]:
    raw = os.getenv("OWNER_USER_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def is_owner(update: Update) -> bool:
    owner_id = get_owner_user_id()
    if owner_id is None:
        return False
    if update.effective_user is None:
        return False
    return int(update.effective_user.id) == owner_id


def is_allowed_chat(update: Update) -> bool:
    allowed = get_allowed_chat_id()
    if allowed is None:
        return True
    if update.effective_chat is None:
        return False
    return int(update.effective_chat.id) == allowed


def delete_collection(coll_ref: firestore.CollectionReference, batch_size: int = 200) -> int:
    deleted = 0
    while True:
        docs = list(coll_ref.limit(batch_size).stream())
        if not docs:
            break
        batch = coll_ref._client.batch()
        for d in docs:
            batch.delete(d.reference)
        batch.commit()
        deleted += len(docs)
    return deleted


async def handle_reset_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_user is None or update.message is None:
        return
    if not is_allowed_chat(update):
        return

    owner_id = get_owner_user_id()
    if owner_id is None or int(update.effective_user.id) != owner_id:
        await update.message.reply_text("권한이 없습니다.")
        return

    text = (update.message.text or "").strip().lower()
    if text != "!reset_db confirm":
        await update.message.reply_text("DB 초기화는 `!RESET_DB CONFIRM` 으로만 가능합니다.")
        return

    db = get_firebase_client()
    chat_id = int(update.effective_chat.id)
    cref = chat_ref(db, chat_id)

    try:
        users_coll = cref.collection("users")
        deleted_users = delete_collection(users_coll)
        cref.delete()
    except Exception:
        await update.message.reply_text("DB 초기화 중 오류가 발생했습니다.")
        return

    await update.message.reply_text(f"DB 초기화 완료. 삭제된 유저 데이터: {deleted_users}개")


async def reset_user_by_username(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target_username: str,
) -> None:
    if update.effective_chat is None or update.message is None:
        return

    if not is_allowed_chat(update):
        return

    if not is_owner(update):
        await update.message.reply_text("권한이 없습니다.")
        return

    chat_id = int(update.effective_chat.id)
    uname = (target_username or "").strip()
    if uname.startswith("@"): 
        uname = uname[1:]
    if not uname:
        await update.message.reply_text("유저명을 확인할 수 없습니다.")
        return

    db = get_firebase_client()
    users_coll = chat_ref(db, chat_id).collection("users")

    docs = list(users_coll.where("username", "==", uname).limit(1).stream())
    if not docs:
        docs = list(users_coll.where("username", "==", uname.lower()).limit(1).stream())

    if not docs:
        await update.message.reply_text(f"@{uname} 유저를 찾을 수 없습니다.")
        return

    dt = now_kst()
    today = kst_date_str(dt)

    udoc = docs[0]
    udoc.reference.set(
        {
            "total_exp": 0,
            "current_level": 1,
            "exp_events": [],
            "exp_gained_today": 0,
            "exp_gained_date": today,
            "warn_count": 0,
            "warn_reset_at": dt + timedelta(hours=24),
            "mute_until": None,
            "mute_tier_today": 0,
            "mute_tier_date": today,
            "last_seen": dt,
        },
        merge=True,
    )

    await update.message.reply_text(f"@{uname} 점수 초기화 완료")


def is_anonymous_admin_message(update: Update) -> bool:
    if update.message is None:
        return False

    if update.message.sender_chat is None:
        return False

    fu = update.message.from_user
    if fu is None:
        return True

    if fu.is_bot and (fu.username == "GroupAnonymousBot" or fu.full_name == "GroupAnonymousBot"):
        return True

    return False


async def handle_exp_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_user is None:
        return

    if not is_allowed_chat(update):
        return

    if is_anonymous_admin_message(update):
        await update.message.reply_text(
            "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
            "익명 관리자 모드를 끄고 다시 `!EXP`를 입력해 주세요."
        )
        return

    if update.effective_chat.type not in (ChatType.SUPERGROUP, ChatType.GROUP):
        return

    db = get_firebase_client()
    chat_id = int(update.effective_chat.id)
    user_id = int(update.effective_user.id)

    async with get_user_lock(chat_id, user_id):
        username = update.effective_user.username
        display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

        dt = now_kst()
        date_key = kst_date_str(dt)

        uref = user_ref(db, chat_id, user_id)

        snap = uref.get()
        data = snap.to_dict() if snap.exists else {}
        uref.set(
            {
                "user_id": user_id,
                "username": username or None,
                "display": display,
                "last_seen": dt,
            },
            merge=True,
        )

        total_exp = int(data.get("total_exp", 0))
        level, progress, need = compute_level(total_exp)
        remaining = need - progress

        users = list(chat_ref(db, chat_id).collection("users").stream())
        rows: List[Tuple[int, int, int]] = []
        for d in users:
            u = d.to_dict() or {}
            tid = int(u.get("user_id", int(d.id)))
            te = int(u.get("total_exp", 0))
            tl = int(u.get("current_level", compute_level(te)[0]))
            rows.append((tid, tl, te))
        rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
        rank = 0
        for i, (tid, _, _) in enumerate(rows, start=1):
            if tid == user_id:
                rank = i
                break
        total_users = len(rows)

    result = {
        "ok": True,
        "level": level,
        "total_exp": total_exp,
        "remaining": remaining,
        "need": need,
        "progress": progress,
        "date": date_key,
    }

    if not result.get("ok"):
        await update.message.reply_text(result["msg"])
        return

    extra_rank = f"\n현재 순위: {rank}/{total_users}" if total_users > 0 else ""
    await update.message.reply_text(
        f"{display}\n"
        f"현재 레벨: Lv.{result['level']}\n"
        f"현재 EXP: {result['total_exp']}\n"
        f"다음 레벨까지 남은 EXP: {result['remaining']}"
        f"{extra_rank}"
    )


async def maybe_delete_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    try:
        await update.message.delete()
    except Exception:
        return


async def kick_user(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> None:
    try:
        await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
        await context.bot.unban_chat_member(chat_id=chat_id, user_id=user_id)
    except Exception:
        return


async def restrict_user(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    until: datetime,
) -> None:
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until,
        )
    except Exception:
        return


def next_mute_minutes(tier_today: int) -> int:
    if tier_today <= 0:
        return 10
    if tier_today == 1:
        return 60
    return 24 * 60


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_user is None:
        return

    if update.effective_user.is_bot or int(update.effective_user.id) == 777000:
        return

    if update.message is None or update.message.text is None:
        return

    if update.effective_chat.type not in (ChatType.SUPERGROUP, ChatType.GROUP):
        return

    if not is_allowed_chat(update):
        return

    text = update.message.text

    if text.strip() == "!가이드":
        await update.message.reply_text(
            "Whalet BOT 명령어 가이드\n"
            "\n"
            "[EXP/레벨]\n"
            "- !exp / .exp: 내 EXP/레벨/순위 확인\n"
            "\n"
            "[출석]\n"
            "- !출첵: 하루 1회 100EXP (출첵 완료 메시지)\n"
            "\n"
            "[메뉴 추천]\n"
            "- !점메추: 점심 메뉴 랜덤 추천\n"
            "- !저메추: 저녁 메뉴 랜덤 추천\n"
            "\n"
            "[덤벼고래 (가위바위보)]\n"
            "- !덤벼고래: 방장에게만 도전 가능한 가위바위보\n"
            "  (하루 2회, 이기면 방장 EXP에서 최대 50EXP 획득)\n"
            "\n"
            "[Based Pals]\n"
            "- !알구매: 100EXP로 알 구매 (1시간 후 부화)\n"
            "- !먹이: 5EXP로 성장치 +5\n"
            "- !마이팔: 내 Pals/알 상태 확인\n"
            "\n"
            "[검 키우기]\n"
            "- !인벤토리: 현재 검/방어티켓 확인\n"
            "- !강화확률: 강화 단계별 비용/확률/판매가 확인\n"
            "- !오른: 강화 진행(확정 버튼)\n"
            "- !당근마켓: 현재 검 판매(확정 버튼)\n"
            "- !베이스드몰: 검 구매(100EXP, 검이 없을 때만 가능)\n"
            "\n"
            "[기타]\n"
            "- !whoami: 내 USER_ID/USERNAME 확인\n"
        )
        return

    if text.strip() == "!알구매":
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!알구매`를 입력해 주세요."
            )
            return

        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        async with get_user_lock(chat_id, user_id):
            db = get_firebase_client()
            dt = now_kst()
            today = kst_date_str(dt)
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}

            if udata.get("pal") or udata.get("egg"):
                await update.message.reply_text("이미 Pals(또는 알)을 보유 중입니다.")
                return

            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

            total_exp = int(udata.get("total_exp", 0))
            if total_exp < PALS_EGG_PRICE_EXP:
                await update.message.reply_text(f"EXP가 부족합니다. (필요 {PALS_EGG_PRICE_EXP}EXP)")
                return

            total_exp -= PALS_EGG_PRICE_EXP
            level = compute_level(total_exp)[0]
            hatch_at = dt + timedelta(hours=1)
            uref.set(
                {
                    "user_id": user_id,
                    "username": username or None,
                    "display": display,
                    "total_exp": total_exp,
                    "current_level": level,
                    "egg": {"hatch_at": hatch_at},
                    "pal": firestore.DELETE_FIELD,
                    "last_seen": dt,
                    "last_active_date": today,
                },
                merge=True,
            )

        egg_url = pals_egg_gif_url()
        msg = (
            f"{display} 님\n"
            f"{PALS_EGG_PRICE_EXP} EXP를 사용해 알을 획득했습니다 🥚\n"
            "1시간 후 부화합니다."
        )
        if egg_url:
            try:
                await update.effective_chat.send_animation(animation=egg_url, caption=msg)
            except Exception:
                await update.message.reply_text(msg)
        else:
            await update.message.reply_text(msg)
        return

    if text.strip() == "!먹이":
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!먹이`를 입력해 주세요."
            )
            return

        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        async with get_user_lock(chat_id, user_id):
            db = get_firebase_client()
            dt = now_kst()
            today = kst_date_str(dt)
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}

            pal = udata.get("pal")
            if not isinstance(pal, dict):
                await update.message.reply_text("현재 Pals가 없습니다. 먼저 `!알구매`로 알을 구매해 주세요.")
                return

            total_exp = int(udata.get("total_exp", 0))
            if total_exp < PALS_FEED_PRICE_EXP:
                await update.message.reply_text(f"EXP가 부족합니다. (필요 {PALS_FEED_PRICE_EXP}EXP)")
                return

            stage = str(pal.get("stage") or "baby")
            type_id = int(pal.get("type_id") or 1)
            growth = int(pal.get("growth") or 0)
            growth += 5
            total_exp -= PALS_FEED_PRICE_EXP

            next_stage = stage
            if stage == "baby" and growth >= PALS_EVOLVE_AT["baby"]:
                next_stage = "teen"
            elif stage == "teen" and growth >= PALS_EVOLVE_AT["teen"]:
                next_stage = "adult"
            elif stage == "adult" and growth >= PALS_EVOLVE_AT["adult"]:
                next_stage = "ultimate"

            level = compute_level(total_exp)[0]
            pal2 = dict(pal)
            pal2["growth"] = growth
            pal2["stage"] = next_stage

            uref.set(
                {
                    "total_exp": total_exp,
                    "current_level": level,
                    "pal": pal2,
                    "last_seen": dt,
                    "last_active_date": today,
                },
                merge=True,
            )

            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

        await update.message.reply_text(
            f"{display} 님\n"
            f"먹이를 주었습니다! (-{PALS_FEED_PRICE_EXP}EXP)\n"
            f"성장치 +5 (현재 {growth})"
        )

        if next_stage != stage:
            img = pals_stage_image_url(next_stage, type_id)
            caption = (
                "✨ 진화 알림\n\n"
                f"{display} 님의 [{pals_display_title(stage, type_id)}]가\n"
                f"[{pals_display_title(next_stage, type_id)}]로 진화했습니다!"
            )
            if img:
                try:
                    await update.effective_chat.send_photo(photo=img, caption=caption)
                except Exception:
                    await update.effective_chat.send_message(caption)
            else:
                await update.effective_chat.send_message(caption)
        return

    if text.strip() == "!마이팔":
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!마이팔`을 입력해 주세요."
            )
            return

        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        async with get_user_lock(chat_id, user_id):
            db = get_firebase_client()
            dt = now_kst()
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}

            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

            egg = udata.get("egg")
            pal = udata.get("pal")

        if isinstance(egg, dict) and egg.get("hatch_at"):
            hatch_at = egg.get("hatch_at")
            remain = int((hatch_at - now_kst()).total_seconds())
            msg = (
                f"{display} 님\n"
                "현재 알을 보유 중입니다 🥚\n"
                f"부화까지 남은 시간: {format_timedelta_kor(remain)}"
            )
            egg_url = pals_egg_gif_url()
            if egg_url:
                try:
                    await update.effective_chat.send_animation(animation=egg_url, caption=msg)
                except Exception:
                    b = await download_url_bytes(egg_url)
                    if b:
                        try:
                            await update.effective_chat.send_animation(
                                animation=InputFile(io.BytesIO(b), filename="palegg.gif"),
                                caption=msg,
                            )
                        except Exception:
                            await update.message.reply_text(msg)
                    else:
                        await update.message.reply_text(msg)
            else:
                base = get_pals_asset_base_url()
                await update.message.reply_text(msg)
            return

        if not isinstance(pal, dict):
            await update.message.reply_text("현재 Pals(또는 알)이 없습니다. `!알구매`로 시작해 주세요.")
            return

        stage = str(pal.get("stage") or "baby")
        type_id = int(pal.get("type_id") or 1)
        growth = int(pal.get("growth") or 0)

        next_need = None
        if stage in PALS_EVOLVE_AT:
            next_need = int(PALS_EVOLVE_AT[stage])
        next_txt = "MAX"
        if next_need is not None:
            next_txt = f"{growth}/{next_need} (남은 {max(0, next_need - growth)})"

        payout = int(PALS_PAYOUT_EXP.get(stage, 0))
        last_payout_at = pal.get("last_payout_at")
        remain_payout_txt = "-"
        if payout > 0 and last_payout_at:
            remain_s = int((last_payout_at + timedelta(hours=24) - now_kst()).total_seconds())
            remain_payout_txt = format_timedelta_kor(remain_s)

        msg = (
            f"{display} 님\n"
            f"[{pals_display_title(stage, type_id)}]\n"
            f"성장치: {growth}\n"
            f"다음 진화: {next_txt}\n"
            f"24h 수익: {payout}EXP\n"
            f"다음 수익까지: {remain_payout_txt}"
        )

        img = pals_stage_image_url(stage, type_id)
        if img:
            try:
                await update.effective_chat.send_photo(photo=img, caption=msg)
            except Exception:
                b = await download_url_bytes(img)
                if b:
                    try:
                        await update.effective_chat.send_photo(
                            photo=InputFile(io.BytesIO(b), filename="pal.png"),
                            caption=msg,
                        )
                    except Exception:
                        await update.message.reply_text(msg)
                else:
                    await update.message.reply_text(msg)
        else:
            base = get_pals_asset_base_url()
            await update.message.reply_text(msg)
        return

    if text.strip() == "!베이스드몰":
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!베이스드몰`을 입력해 주세요."
            )
            return

        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        async with get_user_lock(chat_id, user_id):
            db = get_firebase_client()
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, _ = sword_state_from_udata(udata)
            if lvl != SWORD_NONE_LEVEL:
                await update.message.reply_text("이미 검을 보유 중입니다. (구매는 검이 없을 때만 가능합니다)")
                return

        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="네",
                        callback_data=f"based_mall_buy:{chat_id}:{user_id}:yes",
                    ),
                    InlineKeyboardButton(
                        text="아니오",
                        callback_data=f"based_mall_buy:{chat_id}:{user_id}:no",
                    ),
                ]
            ]
        )
        await update.message.reply_text(
            "검을 구매하시겠습니까? IMF, FTX, 루나, 박상기의 난을 겪은 주인장은 검 당근마켓 판매가격의 20배인 "
            f"{BASED_MALL_PRICE_EXP}EXP에 검을 팔고있습니다.",
            reply_markup=kb,
        )
        return

    if text.strip() in ("!인벤토리", "!inventory"):
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!인벤토리`를 입력해 주세요."
            )
            return
        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        async with get_user_lock(chat_id, user_id):
            db = get_firebase_client()
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, tickets = sword_state_from_udata(udata)
            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))
            if lvl == SWORD_NONE_LEVEL:
                await update.message.reply_text(
                    f"{display}님 현재 검이 없습니다.\n"
                    f"강화 방어티켓: {tickets}장"
                )
            else:
                await update.message.reply_text(
                    f"{display}님 현재소유 검 [{sword_name(lvl)}]이 있습니다.\n"
                    f"강화 방어티켓: {tickets}장"
                )
        return

    if text.strip() == "!강화확률":
        lines: List[str] = []
        for lvl in range(1, SWORD_MAX_LEVEL + 1):
            row = SWORD_TABLE.get(lvl)
            if not row:
                continue
            rate = float(row["rate"]) * 100
            sell = row.get("sell")
            sell_txt = "판매 불가" if sell is None else f"{int(sell)}EXP"
            lines.append(
                f"{lvl}강: {row['name']} | 비용 {int(row['cost'])}EXP | 확률 {rate:.2f}% | 판매가 {sell_txt}"
            )
        await update.message.reply_text("\n".join(lines))
        return

    if text.strip() == "!당근마켓":
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!당근마켓`을 입력해 주세요."
            )
            return
        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        async with get_user_lock(chat_id, user_id):
            db = get_firebase_client()
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, _ = sword_state_from_udata(udata)
            price = sword_sell_price(lvl)
            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))
            if lvl == SWORD_NONE_LEVEL:
                await update.message.reply_text(f"{display}님 현재 검이 없습니다.")
                return
            if price is None:
                await update.message.reply_text(
                    f"{display}님 현재 소유한 [{sword_name(lvl)}]은(는) 판매 불가입니다."
                )
                return
            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text="판매하기",
                            callback_data=f"sword_sell:{chat_id}:{user_id}:yes",
                        ),
                        InlineKeyboardButton(
                            text="취소",
                            callback_data=f"sword_sell:{chat_id}:{user_id}:no",
                        ),
                    ]
                ]
            )
            await update.message.reply_text(
                f"{display}님 현재 소유한 [{sword_name(lvl)}]을 파시겠습니까? 판매가격 {int(price)}EXP",
                reply_markup=kb,
            )
        return

    if text.strip() == "!오른":
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!오른`을 입력해 주세요."
            )
            return
        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        async with get_user_lock(chat_id, user_id):
            db = get_firebase_client()
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, tickets = sword_state_from_udata(udata)
            nxt = sword_next_upgrade_info(lvl)
            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))
            if lvl == SWORD_NONE_LEVEL:
                await update.message.reply_text(f"{display}님 현재 검이 없습니다.")
                return
            if nxt is None:
                await update.message.reply_text(f"{display}님은 이미 최종 검을 보유 중입니다.")
                return
            nxt_level, rate, cost, sell, nxt_name = nxt
            sell_txt = "판매 불가" if sell is None else f"{int(sell)}EXP"
            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text="강화하기",
                            callback_data=f"sword_enhance:{chat_id}:{user_id}:yes",
                        ),
                        InlineKeyboardButton(
                            text="취소",
                            callback_data=f"sword_enhance:{chat_id}:{user_id}:no",
                        ),
                    ]
                ]
            )
            await update.message.reply_text(
                f"{display}님의 [{sword_name(lvl)}]을 강화 하시겠습니까?\n"
                f"강화확률 {rate*100:.2f}%, 강화비용 {int(cost)}exp\n"
                f"강화 후 검[{nxt_name}] 당근마켓 시세 {sell_txt}\n"
                f"보유 방어티켓: {tickets}장",
                reply_markup=kb,
            )
        return

    if text.strip() == "!출첵":
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!출첵`을 입력해 주세요."
            )
            return

        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        async with get_user_lock(chat_id, user_id):
            db = get_firebase_client()
            dt = now_kst()
            today = kst_date_str(dt)
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}

            if udata.get("checkin_date") == today:
                await update.message.reply_text("오늘은 이미 출첵하셨습니다.")
                return

            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

            prev_total = int(udata.get("total_exp", 0))
            new_total = prev_total + 100
            new_level = compute_level(new_total)[0]

            uref.set(
                {
                    "user_id": user_id,
                    "username": username or None,
                    "display": display,
                    "total_exp": new_total,
                    "current_level": new_level,
                    "checkin_date": today,
                    "last_seen": dt,
                    "last_active_date": today,
                },
                merge=True,
            )

        await update.message.reply_text("출첵이 완료되었습니다. 도장 쾅쾅!")
        return

    if text.strip() in ("!점메추", "!저메추"):
        lunch_menu = [
            "김치찌개",
            "된장찌개",
            "제육볶음",
            "비빔밥",
            "돈까스",
            "칼국수",
            "냉면",
            "국밥",
            "초밥",
            "샐러드",
            "햄버거",
            "파스타",
            "쌀국수",
            "피자",
            "라멘",
            "마라탕",
            "떡볶이",
            "치킨",
        ]
        dinner_menu = [
            "삼겹살",
            "곱창",
            "회",
            "치킨",
            "피자",
            "족발",
            "보쌈",
            "찜닭",
            "닭갈비",
            "부대찌개",
            "샤브샤브",
            "카레",
            "스테이크",
            "파스타",
            "타코",
            "중국집(짜장/짬뽕)",
        ]
        if text.strip() == "!점메추":
            pick = random.choice(lunch_menu)
            await update.message.reply_text(f"오늘 점심 추천: {pick}")
        else:
            pick = random.choice(dinner_menu)
            await update.message.reply_text(f"오늘 저녁 추천: {pick}")
        return

    if text.strip() == "!덤벼고래":
        chat_id_for_lock = int(update.effective_chat.id)
        async with get_yacha_chat_lock(chat_id_for_lock):
            dt_now = now_kst()
            today_kst = kst_date_str(dt_now)
            owner_id = get_owner_user_id()
            if owner_id is None:
                await update.message.reply_text("OWNER_USER_ID 설정이 필요합니다.")
                return

            duel0 = get_active_duel(chat_id_for_lock)
            if duel0 and isinstance(duel0, dict):
                created_at0 = duel0.get("created_at")
                accepted0 = bool(duel0.get("accepted"))
                timeout = timedelta(minutes=30) if accepted0 else timedelta(minutes=10)
                if created_at0 and created_at0 < dt_now - timeout:
                    set_active_duel(chat_id_for_lock, None)

            if get_active_duel(chat_id_for_lock) is not None:
                await update.message.reply_text("현재 진행 중인 야차가 있습니다.")
                return

            challenger_id = int(update.effective_user.id)
            if challenger_id == int(owner_id):
                await update.message.reply_text("방장은 자기 자신에게 덤빌 수 없습니다.")
                return

            db = get_firebase_client()
            uref = user_ref(db, chat_id_for_lock, challenger_id)
            snap = uref.get()
            data = snap.to_dict() if snap.exists else {}
            yacha_uses_date = data.get("yacha_uses_date")
            yacha_uses_today = int(data.get("yacha_uses_today", 0))
            if yacha_uses_date != today_kst:
                yacha_uses_date = today_kst
                yacha_uses_today = 0
            if yacha_uses_today >= 2:
                await update.message.reply_text("덤벼고래는 하루 2번만 사용할 수 있습니다.")
                return
            yacha_uses_today += 1
            uref.set(
                {"yacha_uses_date": yacha_uses_date, "yacha_uses_today": yacha_uses_today, "last_seen": dt_now},
                merge=True,
            )

            duel = {
                "chat_id": chat_id_for_lock,
                "challenger_id": challenger_id,
                "challenger_display": f"@{update.effective_user.username}" if update.effective_user.username else str(challenger_id),
                "opponent_id": int(owner_id),
                "opponent_username": None,
                "accepted": False,
                "choices": {},
                "created_at": dt_now,
            }
            set_active_duel(chat_id_for_lock, duel)

            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text="네",
                            callback_data=f"yacha_accept:{chat_id_for_lock}:{duel['challenger_id']}:{duel['opponent_id']}:yes",
                        ),
                        InlineKeyboardButton(
                            text="아니오",
                            callback_data=f"yacha_accept:{chat_id_for_lock}:{duel['challenger_id']}:{duel['opponent_id']}:no",
                        ),
                    ]
                ]
            )
            await update.effective_chat.send_message(
                "방장님, 덤벼고래를 수락하시겠습니까?",
                reply_markup=kb,
            )
            return

    if text.strip() == "!상납금":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return
        context.chat_data["tribute_mode"] = {"step": "await_user_id"}
        await update.message.reply_text("어떤유저에게 싸바싸바를 하시겠습니까?\n유저아이디를 치면(예:XXX)")
        return

    tribute = context.chat_data.get("tribute_mode")
    if is_owner(update) and tribute and isinstance(tribute, dict):
        step = str(tribute.get("step") or "")
        if step == "await_user_id":
            t = text.strip()
            try:
                target_user_id = int(t)
            except ValueError:
                await update.message.reply_text("유저아이디는 숫자로 입력해주세요.")
                return
            tribute["step"] = "await_amount"
            tribute["target_user_id"] = target_user_id
            context.chat_data["tribute_mode"] = tribute
            await update.message.reply_text("얼마의 상납금을 바치시겠습니까?")
            return

        if step == "await_amount":
            t = text.strip()
            try:
                amount = int(t)
            except ValueError:
                await update.message.reply_text("상납금은 숫자로 입력해주세요.")
                return
            if amount <= 0:
                await update.message.reply_text("상납금은 1 이상의 숫자로 입력해주세요.")
                return

            chat_id = int(update.effective_chat.id)
            target_user_id = int(tribute.get("target_user_id") or 0)
            if target_user_id <= 0:
                context.chat_data.pop("tribute_mode", None)
                await update.message.reply_text("대상 유저 정보가 유효하지 않습니다. 다시 `!상납금`부터 진행해주세요.")
                return

            db = get_firebase_client()
            dt = now_kst()
            async with get_user_lock(chat_id, target_user_id):
                uref = user_ref(db, chat_id, target_user_id)
                snap = uref.get()
                udata = snap.to_dict() if snap.exists else {}

                prev_total = int(udata.get("total_exp", 0))
                new_total = prev_total + int(amount)
                new_level = compute_level(new_total)[0]

                target_username = udata.get("username")
                target_display = udata.get("display")
                if not target_display:
                    target_display = f"@{target_username}" if target_username else str(target_user_id)

                uref.set(
                    {
                        "user_id": target_user_id,
                        "username": target_username or None,
                        "display": target_display,
                        "total_exp": new_total,
                        "current_level": new_level,
                        "last_seen": dt,
                    },
                    merge=True,
                )

            context.chat_data.pop("tribute_mode", None)
            owner_name = update.effective_user.full_name if update.effective_user else "방장"
            await update.effective_chat.send_message(
                f"{owner_name}님이 비열하게도 {target_display}님에게 {amount}EXP를 싸바싸바했습니다."
            )
            return

    if text.strip() == "!타노스":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return
        context.chat_data["thanos_mode"] = True
        await update.message.reply_text("타노스할 유저를 선택해주세요.")
        return

    if is_owner(update) and context.chat_data.get("thanos_mode"):
        t = text.strip()
        if t.startswith("!") and len(t) > 1 and " " not in t and t not in (
            "!exp",
            ".exp",
            "!reset_db",
            "!reset_db confirm",
            "!chat_id",
            "!whoami",
            "!리더보드",
            "!leaderboard",
            "!타노스",
        ):
            target_username = t[1:]
            context.chat_data["thanos_mode"] = False
            await reset_user_by_username(update, context, target_username)
            return

    if text.strip() in ("!리더보드", "!leaderboard"):
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return
        await send_leaderboard(context)
        return

    if text.strip().lower() == "!chat_id":
        await update.message.reply_text(f"CHAT_ID: {int(update.effective_chat.id)}")
        return

    if text.strip().lower() == "!whoami":
        u = update.effective_user
        uname = f"@{u.username}" if u.username else (u.full_name or "")
        await update.message.reply_text(f"USER_ID: {int(u.id)}\nUSERNAME: {uname}")
        return

    if text.strip().lower() == "!reset_db confirm" or text.strip().lower() == "!reset_db":
        await handle_reset_db(update, context)
        return

    if is_anonymous_admin_message(update):
        if text.strip().lower() in ("!exp", ".exp"):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!EXP`를 입력해 주세요."
            )
        return

    if text.strip().lower() in ("!exp", ".exp"):
        await handle_exp_query(update, context)
        return

    async with get_user_lock(update.effective_chat.id, update.effective_user.id):
        await _handle_message_locked(update, context)


async def _handle_message_locked(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_user is None:
        return

    if update.message is None or update.message.text is None:
        return

    text = update.message.text

    db = get_firebase_client()
    dt = now_kst()
    today = kst_date_str(dt)

    chat_id = update.effective_chat.id
    chat_title = update.effective_chat.title or None

    user_id = update.effective_user.id
    username = update.effective_user.username
    display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

    cref = chat_ref(db, chat_id)
    uref = user_ref(db, chat_id, user_id)

    contains_url = URL_PATTERN.search(text) is not None

    chat_snap = cref.get()
    user_snap = uref.get()

    cdata = chat_snap.to_dict() if chat_snap.exists else {}
    udata = user_snap.to_dict() if user_snap.exists else {}

    async with get_chat_lock(int(chat_id)):
        chat_snap2 = cref.get()
        cdata2 = chat_snap2.to_dict() if chat_snap2.exists else {}
        counter = int(cdata2.get("blessing_counter", 0))
        defense_counter = int(cdata2.get("defense_counter", 0))
        counter += 1
        defense_counter += 1
        if counter >= 365:
            counter = 0
            total_exp2 = int(udata.get("total_exp", 0)) + 100
            level2 = compute_level(total_exp2)[0]
            uref.set({"total_exp": total_exp2, "current_level": level2}, merge=True)
            await update.effective_chat.send_message(
                f"띠링! 왈렛의 축복이 찾아왔습니다. {display}님이 100EXP를 획득하였습니다."
            )
        if defense_counter >= 500:
            defense_counter = 0
            lvl0, tickets0 = sword_state_from_udata(udata)
            tickets0 += 1
            uref.set({"sword_level": lvl0, "defense_tickets": tickets0}, merge=True)
            await update.effective_chat.send_message(
                f"띠링! 누적채팅 500개를 달성하여 강화 방어티켓을 한장 부여합니다."
            )
        cref.set(
            {
                "chat_id": chat_id,
                "title": chat_title,
                "last_seen": dt,
                "blessing_counter": counter,
                "defense_counter": defense_counter,
            },
            merge=True,
        )

    cur_text = (text or "").strip()

    mute_until = udata.get("mute_until")
    if (not is_owner(update)) and mute_until and mute_until > dt:
        return

    warn_reset_at = udata.get("warn_reset_at")
    warn_count = int(udata.get("warn_count", 0))
    if not warn_reset_at or warn_reset_at <= dt:
        warn_count = 0
        warn_reset_at = dt + timedelta(hours=24)


    mute_tier_date = udata.get("mute_tier_date")
    mute_tier_today = int(udata.get("mute_tier_today", 0))
    if mute_tier_date != today:
        mute_tier_today = 0
        mute_tier_date = today

    if (not is_owner(update)) and contains_url and is_link_block_time(dt):
        warn_count += 1
        warn_reset_at = dt + timedelta(hours=24)

        mute_info: Optional[Dict[str, Any]] = None
        if warn_count >= 3:
            minutes = next_mute_minutes(mute_tier_today)
            mute_tier_today += 1
            warn_count = 0
            mute_until_new = dt + timedelta(minutes=minutes)
            mute_info = {"minutes": minutes, "until": mute_until_new}
            uref.set(
                {
                    "mute_until": mute_until_new,
                    "mute_tier_today": mute_tier_today,
                    "mute_tier_date": mute_tier_date,
                },
                merge=True,
            )

        uref.set(
            {
                "user_id": user_id,
                "username": username or None,
                "display": display,
                "warn_count": warn_count,
                "warn_reset_at": warn_reset_at,
                "mute_tier_today": mute_tier_today,
                "mute_tier_date": mute_tier_date,
                "last_seen": dt,
                "last_active_date": today,
            },
            merge=True,
        )

        cref.set(
            {
                "chat_id": chat_id,
                "title": chat_title,
                "last_seen": dt,
            },
            merge=True,
        )

        await maybe_delete_message(update, context)
        await update.effective_chat.send_message(
            f"⚠️ {display}님 링크 스팸 감지!\n경고 {warn_count}/3"
        )

        if mute_info:
            until = mute_info["until"]
            minutes = mute_info["minutes"]
            await restrict_user(context, chat_id, user_id, until)
            await update.effective_chat.send_message(
                f"🔇 {display}님 경고 누적으로 {minutes}분간 뮤트 처리되었습니다."
            )
        return

    exp_events: List[Dict[str, Any]] = list(udata.get("exp_events", []))
    exp_events = [e for e in exp_events if e.get("ts") and e["ts"] >= dt - timedelta(minutes=1)]
    can_count = len(exp_events) < 3

    gained = 0
    total_exp = int(udata.get("total_exp", 0))
    prev_level = int(udata.get("current_level", compute_level(total_exp)[0]))

    if can_count:
        exp_res = calculate_exp(text, dt)
        gained = exp_res.gained_exp
        if gained > 0:
            exp_events.append({"ts": dt, "exp": gained})
            total_exp += gained

    pal = udata.get("pal")
    if isinstance(pal, dict):
        stage = str(pal.get("stage") or "baby")
        type_id = int(pal.get("type_id") or 1)
        growth = int(pal.get("growth") or 0)
        growth += 5

        next_stage = stage
        if stage == "baby" and growth >= PALS_EVOLVE_AT["baby"]:
            next_stage = "teen"
        elif stage == "teen" and growth >= PALS_EVOLVE_AT["teen"]:
            next_stage = "adult"
        elif stage == "adult" and growth >= PALS_EVOLVE_AT["adult"]:
            next_stage = "ultimate"

        pal2 = dict(pal)
        pal2["growth"] = growth
        pal2["stage"] = next_stage
        udata["pal"] = pal2

    new_level, progress, need = compute_level(total_exp)

    exp_gained_date = udata.get("exp_gained_date")
    exp_gained_today = int(udata.get("exp_gained_today", 0))
    if exp_gained_date != today:
        exp_gained_today = 0
        exp_gained_date = today
    if gained > 0:
        exp_gained_today += gained

    uref.set(
        {
            "user_id": user_id,
            "username": username or None,
            "display": display,
            "total_exp": total_exp,
            "current_level": new_level,
            "exp_events": exp_events,
            "warn_count": warn_count,
            "warn_reset_at": warn_reset_at,
            "last_seen": dt,
            "last_active_date": today,
            "exp_gained_date": exp_gained_date,
            "exp_gained_today": exp_gained_today,
            "pal": udata.get("pal") if isinstance(udata.get("pal"), dict) else firestore.DELETE_FIELD,
        },
        merge=True,
    )

    cref.set(
        {
            "chat_id": chat_id,
            "title": chat_title,
            "last_seen": dt,
        },
        merge=True,
    )

    if new_level != prev_level:
        await update.effective_chat.send_message(
            f"🎉 {display}님 레벨 업!\n현재 레벨 Lv.{new_level}"
        )

    pal_final = udata.get("pal")
    if isinstance(pal_final, dict):
        stage0 = str(pal.get("stage") or "baby") if isinstance(pal, dict) else "baby"
        stage1 = str(pal_final.get("stage") or "baby")
        if stage1 != stage0:
            img = pals_stage_image_url(stage1, int(pal_final.get("type_id") or 1))
            caption = (
                "✨ 진화 알림\n\n"
                f"{display} 님의 [{pals_display_title(stage0, int(pal_final.get('type_id') or 1))}]가\n"
                f"[{pals_display_title(stage1, int(pal_final.get('type_id') or 1))}]로 진화했습니다!"
            )
            if img:
                try:
                    await update.effective_chat.send_photo(photo=img, caption=caption)
                except Exception:
                    await update.effective_chat.send_message(caption)
            else:
                await update.effective_chat.send_message(caption)


async def pals_hatch_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed = get_allowed_chat_id()
    if allowed is None:
        return

    chat_id = int(allowed)
    db = get_firebase_client()
    dt = now_kst()
    today = kst_date_str(dt)

    users = list(chat_ref(db, chat_id).collection("users").stream())
    for udoc in users:
        udata = udoc.to_dict() or {}
        egg = udata.get("egg")
        if not isinstance(egg, dict) or not egg.get("hatch_at"):
            continue
        hatch_at = egg.get("hatch_at")
        if hatch_at > dt:
            continue
        if udata.get("pal"):
            udoc.reference.set({"egg": firestore.DELETE_FIELD}, merge=True)
            continue

        type_id = random.randint(1, 5)
        pal = {
            "stage": "baby",
            "type_id": type_id,
            "growth": 0,
            "hatched_at": dt,
            "last_payout_at": dt,
        }

        display = udata.get("display") or (f"@{udata.get('username')}" if udata.get("username") else str(udoc.id))

        udoc.reference.set(
            {
                "egg": firestore.DELETE_FIELD,
                "pal": pal,
                "last_seen": dt,
                "last_active_date": today,
            },
            merge=True,
        )

        msg = (
            "🐣 부화 알림\n\n"
            f"{display} 님의 알이 부화했습니다!\n"
            f"[{pals_display_title('baby', type_id)}]가 태어났습니다!"
        )
        img = pals_stage_image_url("baby", type_id)
        if img:
            try:
                await context.bot.send_photo(chat_id=chat_id, photo=img, caption=msg)
            except Exception:
                await context.bot.send_message(chat_id=chat_id, text=msg)
        else:
            await context.bot.send_message(chat_id=chat_id, text=msg)


async def pals_payout_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed = get_allowed_chat_id()
    if allowed is None:
        return

    chat_id = int(allowed)
    db = get_firebase_client()
    dt = now_kst()
    today = kst_date_str(dt)

    users = list(chat_ref(db, chat_id).collection("users").stream())
    for udoc in users:
        udata = udoc.to_dict() or {}
        pal = udata.get("pal")
        if not isinstance(pal, dict):
            continue
        stage = str(pal.get("stage") or "baby")
        payout = int(PALS_PAYOUT_EXP.get(stage, 0))
        if payout <= 0:
            continue
        last_payout_at = pal.get("last_payout_at")
        if last_payout_at and last_payout_at > dt - timedelta(hours=24):
            continue

        total_exp = int(udata.get("total_exp", 0)) + payout
        level = compute_level(total_exp)[0]

        pal2 = dict(pal)
        pal2["last_payout_at"] = dt

        udoc.reference.set(
            {
                "total_exp": total_exp,
                "current_level": level,
                "pal": pal2,
                "last_seen": dt,
                "last_active_date": today,
            },
            merge=True,
        )

        display = udata.get("display") or (f"@{udata.get('username')}" if udata.get("username") else str(udoc.id))
        msg = (
            "💰 Pals 수익 알림\n\n"
            f"{display} 님\n"
            f"[{pals_display_title(stage, int(pal.get('type_id') or 1))}]가\n"
            f"오늘의 EXP {payout}을 벌어왔습니다!"
        )
        try:
            await context.bot.send_message(chat_id=chat_id, text=msg)
        except Exception:
            continue


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query is None:
        return

    q = update.callback_query
    data = (q.data or "").strip()
    await q.answer()

    if q.message is None or q.message.chat is None:
        return

    chat_id = int(q.message.chat.id)
    allowed = get_allowed_chat_id()
    if allowed is not None and int(allowed) != chat_id:
        return

    if data.startswith("yacha_accept:"):
        parts = data.split(":")
        if len(parts) == 5:
            _, cid, challenger_id, opponent_id, decision = parts
        elif len(parts) == 6:
            _, _, cid, challenger_id, opponent_id, decision = parts
        else:
            return
        if int(cid) != chat_id:
            return
        duel = get_active_duel(chat_id)
        if duel is None:
            return
        if int(duel.get("challenger_id")) != int(challenger_id) or int(duel.get("opponent_id")) != int(opponent_id):
            return
        if q.from_user is None or int(q.from_user.id) != int(opponent_id):
            return

        if decision == "no":
            set_active_duel(chat_id, None)
            context.chat_data.pop("yacha_pending", None)
            await q.message.edit_text("야차가 거절되었습니다.")
            return

        duel["accepted"] = True
        set_active_duel(chat_id, duel)

        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="가위",
                        callback_data=f"yacha_rps:{chat_id}:{duel['challenger_id']}:{duel['opponent_id']}:scissors",
                    ),
                    InlineKeyboardButton(
                        text="바위",
                        callback_data=f"yacha_rps:{chat_id}:{duel['challenger_id']}:{duel['opponent_id']}:rock",
                    ),
                    InlineKeyboardButton(
                        text="보",
                        callback_data=f"yacha_rps:{chat_id}:{duel['challenger_id']}:{duel['opponent_id']}:paper",
                    ),
                ]
            ]
        )
        await q.message.edit_text(
            "가위 바위 보를 선택하세요. (두 유저의 클릭만 유효)",
            reply_markup=kb,
        )
        return

    if data.startswith("yacha_rps:"):
        parts = data.split(":")
        if len(parts) == 5:
            _, cid, challenger_id, opponent_id, choice = parts
        elif len(parts) == 6:
            _, _, cid, challenger_id, opponent_id, choice = parts
        else:
            return
        if int(cid) != chat_id:
            return
        duel = get_active_duel(chat_id)
        if duel is None or not duel.get("accepted"):
            return
        if int(duel.get("challenger_id")) != int(challenger_id) or int(duel.get("opponent_id")) != int(opponent_id):
            return

        uid = int(q.from_user.id) if q.from_user else 0
        if uid not in (int(challenger_id), int(opponent_id)):
            return
        if choice not in ("rock", "paper", "scissors"):
            return

        choices = duel.get("choices") or {}
        if not isinstance(choices, dict):
            choices = {}
        choices[str(uid)] = choice
        duel["choices"] = choices
        set_active_duel(chat_id, duel)

        if len(choices.keys()) < 2:
            await q.message.edit_text("한 명이 선택했습니다. 상대의 선택을 기다리는 중...", reply_markup=q.message.reply_markup)
            return

        a_id = int(challenger_id)
        b_id = int(opponent_id)
        a_choice = str(choices.get(str(a_id)))
        b_choice = str(choices.get(str(b_id)))

        await q.message.edit_text("3...")
        await asyncio.sleep(1)
        await q.message.edit_text("2...")
        await asyncio.sleep(1)
        await q.message.edit_text("1...")
        await asyncio.sleep(1)

        res = rps_result(a_choice, b_choice)
        if res == 0:
            set_active_duel(chat_id, None)
            await q.message.edit_text("비겼습니다. 야차 종료!")
            return

        winner_id = a_id if res == 1 else b_id
        loser_id = b_id if res == 1 else a_id

        challenger_display = str(duel.get("challenger_display") or str(a_id))
        owner_id = get_owner_user_id()
        if owner_id is not None and int(owner_id) == int(b_id):
            opponent_display = "방장"
        else:
            opponent_display = f"@{str(duel.get('opponent_username') or '')}" if duel.get("opponent_username") else str(b_id)
        winner_display = challenger_display if winner_id == a_id else opponent_display
        loser_display = opponent_display if winner_id == a_id else challenger_display

        db = get_firebase_client()
        delta = 0
        if owner_id is not None and int(owner_id) in (a_id, b_id):
            challenger_id_int = b_id if int(owner_id) == a_id else a_id
            owner_id_int = int(owner_id)
            if winner_id == challenger_id_int:
                lock1, lock2 = await acquire_two_user_locks(chat_id, owner_id_int, challenger_id_int)
                try:
                    oref = user_ref(db, chat_id, owner_id_int)
                    cref = user_ref(db, chat_id, challenger_id_int)
                    osnap = oref.get()
                    csnap = cref.get()
                    odata = osnap.to_dict() if osnap.exists else {}
                    cdata = csnap.to_dict() if csnap.exists else {}
                    oexp = int(odata.get("total_exp", 0))
                    cexp = int(cdata.get("total_exp", 0))
                    delta = min(50, max(0, oexp))
                    oexp2 = max(0, oexp - delta)
                    cexp2 = cexp + delta
                    olevel2 = compute_level(oexp2)[0]
                    clevel2 = compute_level(cexp2)[0]
                    oref.set({"total_exp": oexp2, "current_level": olevel2}, merge=True)
                    cref.set({"total_exp": cexp2, "current_level": clevel2}, merge=True)
                finally:
                    release_two_user_locks(lock1, lock2)

        set_active_duel(chat_id, None)
        transfer_line = (
            f"EXP 이체: {loser_display} → {winner_display} ({delta} EXP)"
            if delta > 0
            else "EXP 이체: 없음"
        )
        await q.message.edit_text(f"결과: {winner_display} 승!\n{transfer_line}")
        return

    if data.startswith("based_mall_buy:"):
        parts = data.split(":")
        if len(parts) != 4:
            return
        _, cid, uid, decision = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            try:
                await q.answer("명령어를 친 본인만 누를 수 있습니다.", show_alert=True)
            except Exception:
                return
            return
        if decision != "yes":
            await q.message.edit_text("구매가 취소되었습니다.")
            return

        db = get_firebase_client()
        target_user_id = int(uid)
        async with get_user_lock(chat_id, target_user_id):
            uref = user_ref(db, chat_id, target_user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, tickets = sword_state_from_udata(udata)
            if lvl != SWORD_NONE_LEVEL:
                await q.message.edit_text("이미 검을 보유 중입니다.")
                return

            total_exp = int(udata.get("total_exp", 0))
            if total_exp < BASED_MALL_PRICE_EXP:
                await q.message.edit_text(f"EXP가 부족합니다. (필요 {BASED_MALL_PRICE_EXP}EXP)")
                return

            total_exp -= BASED_MALL_PRICE_EXP
            new_level = compute_level(total_exp)[0]
            uref.set(
                {
                    "total_exp": total_exp,
                    "current_level": new_level,
                    "sword_level": BASED_MALL_SWORD_LEVEL,
                    "defense_tickets": tickets,
                },
                merge=True,
            )

        await q.message.edit_text(
            f"구매 완료! [{sword_name(BASED_MALL_SWORD_LEVEL)}] 지급 완료. (-{BASED_MALL_PRICE_EXP}EXP)"
        )
        return

    if data.startswith("sword_sell:"):
        parts = data.split(":")
        if len(parts) != 4:
            return
        _, cid, uid, decision = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            try:
                await q.answer("명령어를 친 본인만 누를 수 있습니다.", show_alert=True)
            except Exception:
                return
            return
        if decision != "yes":
            await q.message.edit_text("판매가 취소되었습니다.")
            return

        db = get_firebase_client()
        target_user_id = int(uid)
        async with get_user_lock(chat_id, target_user_id):
            uref = user_ref(db, chat_id, target_user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, tickets = sword_state_from_udata(udata)
            if lvl == SWORD_NONE_LEVEL:
                await q.message.edit_text("현재 검이 없습니다.")
                return
            price = sword_sell_price(lvl)
            if price is None:
                await q.message.edit_text("현재 검은 판매 불가입니다.")
                return
            prev_total = int(udata.get("total_exp", 0))
            new_total = prev_total + int(price)
            new_level = compute_level(new_total)[0]
            uref.set(
                {
                    "total_exp": new_total,
                    "current_level": new_level,
                    "sword_level": SWORD_NONE_LEVEL,
                    "defense_tickets": tickets,
                },
                merge=True,
            )
        await q.message.edit_text(f"판매 완료! {int(price)}EXP를 획득했습니다.\n현재 검: 없음")
        return

    if data.startswith("sword_enhance:"):
        parts = data.split(":")
        if len(parts) != 4:
            return
        _, cid, uid, decision = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            try:
                await q.answer("명령어를 친 본인만 누를 수 있습니다.", show_alert=True)
            except Exception:
                return
            return
        if decision != "yes":
            await q.message.edit_text("강화가 취소되었습니다.")
            return

        db = get_firebase_client()
        target_user_id = int(uid)
        async with get_user_lock(chat_id, target_user_id):
            uref = user_ref(db, chat_id, target_user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, tickets = sword_state_from_udata(udata)
            if lvl == SWORD_NONE_LEVEL:
                await q.message.edit_text("현재 검이 없습니다.")
                return
            nxt = sword_next_upgrade_info(lvl)
            if nxt is None:
                await q.message.edit_text("이미 최종 검입니다.")
                return
            nxt_level, rate, cost, _, nxt_name = nxt

            total_exp = int(udata.get("total_exp", 0))
            if total_exp < int(cost):
                await q.message.edit_text(f"EXP가 부족합니다. (필요 {int(cost)}EXP)")
                return

            total_exp -= int(cost)
            success = random.random() < float(rate)
            if success:
                lvl2 = nxt_level
                msg = f"강화 성공! [{nxt_name}] 획득!"
            else:
                if tickets > 0:
                    tickets -= 1
                    lvl2 = lvl
                    msg = "강화 실패! 방어티켓 1장을 사용하여 검이 복구되었습니다."
                else:
                    lvl2 = SWORD_NONE_LEVEL
                    msg = "강화 실패! 검이 파괴되어 사라졌습니다."

            new_level = compute_level(total_exp)[0]
            uref.set(
                {
                    "total_exp": total_exp,
                    "current_level": new_level,
                    "sword_level": lvl2,
                    "defense_tickets": tickets,
                },
                merge=True,
            )

        if lvl2 == SWORD_NONE_LEVEL:
            await q.message.edit_text(f"{msg}\n남은 방어티켓: {tickets}장")
        else:
            await q.message.edit_text(
                f"{msg}\n현재 검: [{sword_name(lvl2)}]\n남은 방어티켓: {tickets}장"
            )
        return


async def send_fever_start(context: ContextTypes.DEFAULT_TYPE) -> None:
    db = get_firebase_client()
    dt = now_kst()

    allowed = get_allowed_chat_id()
    if allowed is not None:
        try:
            await context.bot.send_message(
                chat_id=int(allowed),
                text=(
                    "🔥 피버타임이 적용됩니다!\n"
                    "지금부터 오후 11시까지 모든 EXP 획득량 1.5배입니다."
                ),
            )
        except Exception:
            return
        return

    chats = db.collection("chats").stream()
    for doc in chats:
        data = doc.to_dict() or {}
        chat_id = data.get("chat_id")
        if not chat_id:
            continue
        try:
            await context.bot.send_message(
                chat_id=int(chat_id),
                text=(
                    "🔥 피버타임이 적용됩니다!\n"
                    "지금부터 오후 11시까지 모든 EXP 획득량 1.5배입니다."
                ),
            )
        except Exception:
            continue


async def send_fever_end(context: ContextTypes.DEFAULT_TYPE) -> None:
    db = get_firebase_client()
    allowed = get_allowed_chat_id()
    if allowed is not None:
        try:
            await context.bot.send_message(
                chat_id=int(allowed),
                text=(
                    "🧊 피버타임 종료!\n"
                    "이제부터 EXP는 기본 배율로 돌아갑니다."
                ),
            )
        except Exception:
            return
        return

    chats = db.collection("chats").stream()
    for doc in chats:
        data = doc.to_dict() or {}
        chat_id = data.get("chat_id")
        if not chat_id:
            continue
        try:
            await context.bot.send_message(
                chat_id=int(chat_id),
                text=(
                    "🧊 피버타임 종료!\n"
                    "이제부터 EXP는 기본 배율로 돌아갑니다."
                ),
            )
        except Exception:
            continue


def ordinal_emoji(n: int) -> str:
    return {1: "1️⃣", 2: "2️⃣", 3: "3️⃣"}.get(n, f"{n}️⃣")


async def send_leaderboard(context: ContextTypes.DEFAULT_TYPE) -> None:
    db = get_firebase_client()
    dt = now_kst()
    now_label = dt.strftime("%H:%M")
    today = kst_date_str(dt)
    fever = is_fever_time(dt)

    allowed = get_allowed_chat_id()
    chats: List[Any]
    if allowed is not None:
        chats = [db.collection("chats").document(str(int(allowed))).get()]
        chats = [c for c in chats if c.exists]
    else:
        chats = list(db.collection("chats").stream())

    for cdoc in chats:
        cdata = cdoc.to_dict() or {}
        chat_id = cdata.get("chat_id")
        if not chat_id:
            continue

        users = list(cdoc.reference.collection("users").stream())
        user_rows: List[Dict[str, Any]] = []
        active_today = 0
        exp_today = 0

        for udoc in users:
            udata = udoc.to_dict() or {}
            total_exp = int(udata.get("total_exp", 0))
            level = int(udata.get("current_level", compute_level(total_exp)[0]))
            display = udata.get("display") or udata.get("username") or udoc.id
            if isinstance(display, str) and display.startswith("@"):
                display = display[1:]

            user_rows.append(
                {
                    "display": display,
                    "level": level,
                    "exp": total_exp,
                }
            )

            if udata.get("last_active_date") == today:
                active_today += 1
            if udata.get("exp_gained_date") == today and int(udata.get("exp_gained_today", 0)) > 0:
                exp_today += 1

        user_rows.sort(key=lambda x: (x["level"], x["exp"]), reverse=True)

        top3 = user_rows[:10]
        lines = [f"🏆 Whalet CHAT LEADERBOARD ({now_label})", ""]

        for i, row in enumerate(top3, start=1):
            fire = " 🔥" if fever else ""
            lines.append(
                f"{ordinal_emoji(i)} {row['display']} | Lv.{row['level']} | {row['exp']} EXP{fire}"
            )

        if len(user_rows) >= 2:
            gap = int(user_rows[0]["exp"]) - int(user_rows[1]["exp"])
            gap = abs(gap)
            lines.append("")
            lines.append(f"⚡ 다음 순위까지 단 {gap} EXP 차이!")

        if user_rows:
            top_n = max(1, int((len(user_rows) * 0.1) + 0.9999))
            top_levels = [r["level"] for r in user_rows[:top_n]]
            avg_level = sum(top_levels) / len(top_levels)
            lines.append("")
            lines.append(f"📌 현재 상위 10% 평균 레벨: Lv.{int(round(avg_level))}")

        if active_today > 0:
            pct = int(round((exp_today / active_today) * 100))
            lines.append(f"📌 지금 활동 유저 중 {pct}%가 오늘 EXP 획득")

        text = "\n".join(lines)

        try:
            await context.bot.send_message(chat_id=int(chat_id), text=text)
        except Exception:
            continue


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
 
    application = Application.builder().token(token).build()

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_callback))
 
    from zoneinfo import ZoneInfo
 
    kst = ZoneInfo(KST_TZ)
    application.job_queue.run_daily(send_fever_start, time=time(19, 0, tzinfo=kst))
    application.job_queue.run_daily(send_fever_end, time=time(23, 0, tzinfo=kst))
    application.job_queue.run_daily(send_leaderboard, time=time(10, 0, tzinfo=kst))
    application.job_queue.run_daily(send_leaderboard, time=time(14, 0, tzinfo=kst))
    application.job_queue.run_daily(send_leaderboard, time=time(18, 0, tzinfo=kst))
    application.job_queue.run_daily(send_leaderboard, time=time(22, 0, tzinfo=kst))

    application.job_queue.run_repeating(pals_hatch_job, interval=60, first=10)
    application.job_queue.run_repeating(pals_payout_job, interval=300, first=30)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
