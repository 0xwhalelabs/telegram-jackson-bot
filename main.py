import asyncio
import base64
import hashlib
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
try:
    from google.cloud.firestore_v1 import FieldFilter
except Exception:
    from google.cloud.firestore_v1.base_query import FieldFilter
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


_RR_GAMES: Dict[int, Dict[str, Any]] = {}


_YACHA_CHAT_LOCKS: Dict[int, asyncio.Lock] = {}


_RR_CHAT_LOCKS: Dict[int, asyncio.Lock] = {}


_CHAT_LOCKS: Dict[int, asyncio.Lock] = {}


_FISHING_SESSIONS: Dict[Tuple[int, int], bool] = {}
_FISHING_PENDING: Dict[Tuple[int, int], bool] = {}


def get_chat_lock(chat_id: int) -> asyncio.Lock:
    key = int(chat_id)
    lock = _CHAT_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _CHAT_LOCKS[key] = lock
    return lock


def get_rr_chat_lock(chat_id: int) -> asyncio.Lock:
    key = int(chat_id)
    lock = _RR_CHAT_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _RR_CHAT_LOCKS[key] = lock
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

KEYWORD_PATTERN = re.compile(r"(?i)(\bbased\b)")
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


def mask_treasure_hint(cmd: str) -> str:
    keep = set(["!", ","])
    chars = list(cmd)
    idxs = [i for i, ch in enumerate(chars) if ch not in keep]
    if not idxs:
        return cmd
    reveal_cnt = max(1, int(round(len(idxs) * 0.35)))
    reveal_cnt = min(reveal_cnt, len(idxs))
    reveal = set(random.sample(idxs, reveal_cnt))
    out: List[str] = []
    for i, ch in enumerate(chars):
        if ch in keep:
            out.append(ch)
        elif i in reveal:
            out.append(ch)
        else:
            out.append("☆")
    return "".join(out)


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
    return ExpResult(20, "fixed")


def _normalize_username(uname: str) -> str:
    v = (uname or "").strip()
    if v.startswith("@"): 
        v = v[1:]
    return v.strip()


def _find_user_doc_by_username(db: firestore.Client, chat_id: int, username: str):
    uname = _normalize_username(username)
    if not uname:
        return None
    users_coll = chat_ref(db, int(chat_id)).collection("users")
    docs = list(users_coll.where(filter=FieldFilter("username", "==", uname)).limit(1).stream())
    if not docs:
        docs = list(users_coll.where(filter=FieldFilter("username", "==", uname.lower())).limit(1).stream())
    if not docs:
        return None
    return docs[0]


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


def get_active_rr(chat_id: int) -> Optional[Dict[str, Any]]:
    return _RR_GAMES.get(int(chat_id))


def set_active_rr(chat_id: int, game: Optional[Dict[str, Any]]) -> None:
    key = int(chat_id)
    if game is None:
        _RR_GAMES.pop(key, None)
    else:
        _RR_GAMES[key] = game


def user_link(user_id: int, label: str) -> str:
    uid = int(user_id)
    name = (label or str(uid)).replace("<", "").replace(">", "")
    return f"<a href=\"tg://user?id={uid}\">{name}</a>"


def rr_invite_job_name(chat_id: int, message_id: int) -> str:
    return f"rr_invite_timeout:{int(chat_id)}:{int(message_id)}"


def rr_action_job_name(chat_id: int, message_id: int) -> str:
    return f"rr_action_tick:{int(chat_id)}:{int(message_id)}"


def rr_cancel_jobs(context: ContextTypes.DEFAULT_TYPE, game: Dict[str, Any]) -> None:
    try:
        chat_id = int(game.get("chat_id") or 0)
        message_id = int(game.get("message_id") or 0)
    except Exception:
        return

    if chat_id <= 0 or message_id <= 0:
        return
    try:
        for j in context.job_queue.get_jobs_by_name(rr_invite_job_name(chat_id, message_id)):
            j.schedule_removal()
    except Exception:
        pass
    try:
        for j in context.job_queue.get_jobs_by_name(rr_action_job_name(chat_id, message_id)):
            j.schedule_removal()
    except Exception:
        pass


async def rr_set_message(
    context: ContextTypes.DEFAULT_TYPE,
    game: Dict[str, Any],
    base_text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
    countdown: Optional[int] = None,
    edit_existing: bool = False,
) -> None:
    chat_id = int(game.get("chat_id") or 0)
    message_id = int(game.get("message_id") or 0)
    if chat_id <= 0:
        return

    game["status_text"] = base_text
    game["last_reply_markup"] = reply_markup
    if countdown is not None:
        game["countdown_until"] = now_kst() + timedelta(seconds=int(countdown))
    else:
        game.pop("countdown_until", None)

    text = base_text
    if countdown is not None:
        text = f"{base_text}\n\n남은 시간: {int(countdown)}초"

    if edit_existing and message_id > 0:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
            return
        except Exception:
            pass

    try:
        sent = await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        game["message_id"] = int(sent.message_id)
        set_active_rr(chat_id, game)
    except Exception:
        return


def rr_start_invite_timeout(context: ContextTypes.DEFAULT_TYPE, game: Dict[str, Any]) -> None:
    rr_cancel_jobs(context, game)
    chat_id = int(game.get("chat_id") or 0)
    message_id = int(game.get("message_id") or 0)
    if chat_id <= 0 or message_id <= 0:
        return
    context.job_queue.run_once(
        rr_invite_timeout_job,
        when=300,
        name=rr_invite_job_name(chat_id, message_id),
        data={"chat_id": chat_id, "message_id": message_id},
    )


def rr_start_action_timeout(
    context: ContextTypes.DEFAULT_TYPE, game: Dict[str, Any], required_user_ids: List[int]
) -> None:
    rr_cancel_jobs(context, game)
    chat_id = int(game.get("chat_id") or 0)
    message_id = int(game.get("message_id") or 0)
    if chat_id <= 0 or message_id <= 0:
        return
    game["required_user_ids"] = [int(x) for x in required_user_ids]
    game["countdown_until"] = now_kst() + timedelta(seconds=30)
    context.job_queue.run_repeating(
        rr_action_tick_job,
        interval=1,
        first=0,
        name=rr_action_job_name(chat_id, message_id),
        data={"chat_id": chat_id, "message_id": message_id},
    )


async def rr_invite_timeout_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data or {}
    chat_id = int(data.get("chat_id") or 0)
    message_id = int(data.get("message_id") or 0)
    if chat_id <= 0 or message_id <= 0:
        return

    async with get_rr_chat_lock(chat_id):
        game = get_active_rr(chat_id)
        if not game:
            return
        if int(game.get("message_id") or 0) != message_id:
            return
        if game.get("phase") != "invite":
            return
        set_active_rr(chat_id, None)

    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text="상대가 5분 안에 수락하지 않아 러시안룰렛이 취소되었습니다.",
        )
    except Exception:
        return


async def rr_action_tick_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data or {}
    chat_id = int(data.get("chat_id") or 0)
    if chat_id <= 0:
        return

    game = get_active_rr(chat_id)
    if not game:
        context.job.schedule_removal()
        return
    cur_mid = int(game.get("message_id") or 0)
    if cur_mid <= 0:
        context.job.schedule_removal()
        return
    until = game.get("countdown_until")
    if not until:
        context.job.schedule_removal()
        return
    remaining = int((until - now_kst()).total_seconds())

    base_text = str(game.get("status_text") or "")
    phase = str(game.get("phase") or "")
    required = game.get("required_user_ids")
    if not isinstance(required, list):
        required = []

    reply_markup = game.get("last_reply_markup")
    if remaining > 0:
        text = f"{base_text}\n\n남은 시간: {remaining}초"
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=cur_mid,
                text=text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
        except Exception:
            pass
        return

    context.job.schedule_removal()

    forfeiter_id = 0
    if phase == "rps":
        choices = game.get("rps_choices")
        if not isinstance(choices, dict):
            choices = {}
        for uid in required:
            if str(int(uid)) not in choices:
                forfeiter_id = int(uid)
                break
    elif phase == "order":
        forfeiter_id = int(game.get("winner_id") or 0)
    elif phase == "pick":
        forfeiter_id = int(game.get("turn_id") or 0)

    c_id = int(game.get("challenger_id") or 0)
    o_id = int(game.get("opponent_id") or 0)
    other_id = o_id if forfeiter_id == c_id else c_id
    pot = int(game.get("pot") or 0)

    if forfeiter_id and other_id and pot > 0:
        db = get_firebase_client()
        dt = now_kst()
        lock1, lock2 = await acquire_two_user_locks(chat_id, c_id, o_id)
        try:
            sref = user_ref(db, chat_id, other_id)
            ssnap = sref.get()
            sudata = ssnap.to_dict() if ssnap.exists else {}
            s_bal = int(sudata.get("total_exp", 0)) + pot
            sref.set({"total_exp": s_bal, "last_seen": dt}, merge=True)
        finally:
            release_two_user_locks(lock1, lock2)

    fdisp = user_link(
        forfeiter_id,
        str(game.get("challenger_display") if forfeiter_id == c_id else game.get("opponent_display")),
    )
    sdisp = user_link(
        other_id,
        str(game.get("challenger_display") if other_id == c_id else game.get("opponent_display")),
    )
    set_active_rr(chat_id, None)
    text = (
        f"시간 초과! {fdisp}님이 30초 내 선택하지 않아 기권패 처리되었습니다.\n"
        f"{sdisp}님은 전리품으로 {pot}$WHAT을 획득하셨습니다."
    )
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
        )
    except Exception:
        pass


def rps_result(a_choice: str, b_choice: str) -> int:
    beats = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    if a_choice == b_choice:
        return 0
    if beats.get(a_choice) == b_choice:
        return 1
    return -1


PALS_EGG_PRICE_EXP = 100


FISHING_TRASH_ITEMS: List[str] = [
    "비탈릭의 휴지뭉치",
    "부셔진 루나코인조각",
    "샘 뱅크먼의 호소문",
    "구겨진 밈코인 전단지",
    "테더 영수증 찢어진 조각",
    "잭슨의 미확인 수상한 USB",
    "KOL의 눈물 젖은 DM",
    "0.00000001BTC 적힌 수첩",
]


FISHING_COMMON_FISH: List[str] = [
    "숭어",
    "장어",
    "복어",
    "광어",
    "놀래미",
    "붕어",
    "우럭",
    "도다리",
    "고등어",
    "멸치",
]


FISHING_RARE_FISH: List[str] = [
    "황금 참치",
    "심해 아귀",
    "전설의 철갑상어",
    "레어 블루랍스터",
    "유니콘 해마",
]


FISHING_SATOSHI_NOTE = "사토시의 비밀노트"


FISHING_WAIT_MESSAGES: List[str] = [
    "찌가 꿈틀대는 느낌이 듭니다...",
    "수면 위로 수상한 물결이 일렁입니다.",
    "옆 사람 낚싯줄과 얽힐 뻔했습니다.",
    "바닷바람이 쎄게 붑니다.",
    "미끼가 뭔가에 뜯긴 것 같습니다.",
    "갑자기 고래가 지나간 것 같습니다.",
    "낚싯대를 꽉 잡으세요!",
    "어딘가에서 '펌프잇'이 들려옵니다.",
]


def fishing_daily_limit(tool_level: int) -> int:
    lvl = max(0, int(tool_level))
    return 10 + (lvl * 2)


def _get_fishing_session_key(chat_id: int, user_id: int) -> Tuple[int, int]:
    return int(chat_id), int(user_id)


def is_fishing_active(chat_id: int, user_id: int) -> bool:
    return bool(_FISHING_SESSIONS.get(_get_fishing_session_key(chat_id, user_id)))


def set_fishing_active(chat_id: int, user_id: int, active: bool) -> None:
    key = _get_fishing_session_key(chat_id, user_id)
    if not active:
        _FISHING_SESSIONS.pop(key, None)
        return
    _FISHING_SESSIONS[key] = True


def _get_fishing_pending_key(chat_id: int, user_id: int) -> Tuple[int, int]:
    return int(chat_id), int(user_id)


def is_fishing_pending(chat_id: int, user_id: int) -> bool:
    return bool(_FISHING_PENDING.get(_get_fishing_pending_key(chat_id, user_id)))


def set_fishing_pending(chat_id: int, user_id: int, pending: bool) -> None:
    key = _get_fishing_pending_key(chat_id, user_id)
    if not pending:
        _FISHING_PENDING.pop(key, None)
        return
    _FISHING_PENDING[key] = True


def fish_cast_job_name(chat_id: int, user_id: int, message_id: int) -> str:
    return f"fish_cast_delay:{int(chat_id)}:{int(user_id)}:{int(message_id)}"


def fish_cancel_jobs(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, message_id: int) -> None:
    try:
        for j in context.job_queue.get_jobs_by_name(fish_cast_job_name(int(chat_id), int(user_id), int(message_id))):
            j.schedule_removal()
    except Exception:
        pass


def _fishing_kb(chat_id: int, user_id: int, message_id: int, can_continue: bool) -> InlineKeyboardMarkup:
    row: List[InlineKeyboardButton] = []
    if can_continue:
        row.append(
            InlineKeyboardButton(
                text="계속하기",
                callback_data=f"fish_cast:{int(chat_id)}:{int(user_id)}:{int(message_id)}",
            )
        )
    row.append(
        InlineKeyboardButton(
            text="끝내기",
            callback_data=f"fish_end:{int(chat_id)}:{int(user_id)}:{int(message_id)}",
        )
    )
    return InlineKeyboardMarkup([row])


async def fish_cast_delayed_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data if context.job else None
    if not isinstance(data, dict):
        return
    chat_id = int(data.get("chat_id") or 0)
    user_id = int(data.get("user_id") or 0)
    message_id = int(data.get("message_id") or 0)
    username = data.get("username")
    display = str(data.get("display") or str(user_id))
    if chat_id <= 0 or user_id <= 0 or message_id <= 0:
        return

    if not is_fishing_active(chat_id, user_id):
        set_fishing_pending(chat_id, user_id, False)
        return

    try:
        db = get_firebase_client()
        dt = now_kst()
        res = await _do_fishing_cast(db, chat_id, user_id, username, display, dt)
    except Exception:
        set_fishing_pending(chat_id, user_id, False)
        msg = "낚시 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=msg,
            )
        except Exception:
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Exception:
                pass
        return

    if not bool(res.get("ok")):
        set_fishing_active(chat_id, user_id, False)
        set_fishing_pending(chat_id, user_id, False)
        msg = str(res.get("msg") or "낚시에 실패했습니다.")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=msg,
            )
        except Exception:
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Exception:
                pass
        return

    remaining = int(res.get("remaining") or 0)
    limit = int(res.get("limit") or 0)
    loot_name = str(res.get("loot_name") or "")
    loot_value = int(res.get("loot_value") or 0)
    price_line = str(res.get("price_line") or "").strip()

    can_continue = remaining > 0
    if not can_continue:
        set_fishing_active(chat_id, user_id, False)

    set_fishing_pending(chat_id, user_id, False)

    text = (
        f"{display} 낚시!\n"
        f"획득: {loot_name}\n"
        f"가치: {loot_value}$WHAT\n"
        + (price_line + "\n" if price_line else "")
        + f"남은 횟수: {remaining}/{limit}"
    )
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=_fishing_kb(chat_id, user_id, message_id, can_continue=can_continue),
        )
    except Exception:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=_fishing_kb(chat_id, user_id, message_id, can_continue=can_continue),
            )
        except Exception:
            pass


def _coerce_int_dict(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            continue
        try:
            out[k] = int(v)
        except Exception:
            continue
    return out


async def _ensure_daily_fish_prices(
    db: firestore.Client, chat_id: int, dt: datetime, force: bool = False
) -> Dict[str, int]:
    today = kst_date_str(dt)
    async with get_chat_lock(int(chat_id)):
        cref = chat_ref(db, int(chat_id))
        csnap = cref.get()
        cdata = csnap.to_dict() if csnap.exists else {}

        if not force and cdata.get("fish_market_date") == today:
            prices = cdata.get("fish_prices")
            if isinstance(prices, dict) and prices:
                return {k: int(v) for k, v in prices.items() if isinstance(k, str)}

        prices2: Dict[str, int] = {}
        for name in FISHING_COMMON_FISH:
            prices2[name] = random.randint(80, 150)
        for name in FISHING_RARE_FISH:
            prices2[name] = random.randint(500, 800)
        prices2[FISHING_SATOSHI_NOTE] = 100_000

        cref.set(
            {
                "chat_id": int(chat_id),
                "fish_market_date": today,
                "fish_prices": prices2,
                "last_seen": dt,
            },
            merge=True,
        )

        return prices2


async def _do_fishing_cast(
    db: firestore.Client,
    chat_id: int,
    user_id: int,
    username: Optional[str],
    display: str,
    dt: datetime,
) -> Dict[str, Any]:
    today = kst_date_str(dt)
    async with get_user_lock(int(chat_id), int(user_id)):
        uref = user_ref(db, int(chat_id), int(user_id))
        snap = uref.get()
        udata = snap.to_dict() if snap.exists else {}

        sword_lvl, _ = sword_state_from_udata(udata)
        rod_lvl = udata.get("fishing_rod_level")
        if rod_lvl is None:
            return {
                "ok": False,
                "msg": f"{display} 낚시는 낚싯대가 있어야 가능합니다.\n!오른 에서 검→낚싯대 교환을 먼저 해주세요.",
            }

        tool_lvl = int(rod_lvl)
        if tool_lvl < 0:
            tool_lvl = 0

        limit = fishing_daily_limit(tool_lvl)
        uses_date = udata.get("fishing_uses_date")
        uses_today = int(udata.get("fishing_uses_today", 0))
        if uses_date != today:
            uses_today = 0
            uses_date = today
        remaining = max(0, limit - uses_today)
        if remaining <= 0:
            return {
                "ok": False,
                "msg": f"{display}님 오늘 낚시 가능 횟수를 모두 사용했습니다. (하루 {limit}회)",
                "limit": limit,
                "remaining": 0,
            }

        prices = await _ensure_daily_fish_prices(db, int(chat_id), dt)
        fish_inv = _coerce_int_dict(udata.get("fish_inventory"))
        note_cnt = int(udata.get("satoshi_note", 0))
        tickets_list, _ = defense_tickets_list_from_udata(udata, dt)

        weights = [80.0, 40.0, 20.0, 1.0, 0.05]
        pick = random.random() * sum(weights)
        cat = "trash"
        acc = 0.0
        for name, w in zip(["trash", "common", "rare", "ticket", "note"], weights):
            acc += float(w)
            if pick <= acc:
                cat = name
                break

        loot_name = ""
        loot_value = 0
        if cat == "trash":
            loot_name = random.choice(FISHING_TRASH_ITEMS)
            loot_value = 0
        elif cat == "common":
            loot_name = random.choice(FISHING_COMMON_FISH)
            fish_inv[loot_name] = int(fish_inv.get(loot_name, 0)) + 1
            loot_value = int(prices.get(loot_name, 0))
        elif cat == "rare":
            loot_name = random.choice(FISHING_RARE_FISH)
            fish_inv[loot_name] = int(fish_inv.get(loot_name, 0)) + 1
            loot_value = int(prices.get(loot_name, 0))
        elif cat == "ticket":
            loot_name = "강화 방어권"
            tickets_list.append(dt + timedelta(seconds=DEFENSE_TICKET_TTL_SECONDS))
            loot_value = random.randint(5000, 8000)
        else:
            loot_name = FISHING_SATOSHI_NOTE
            note_cnt += 1
            loot_value = 100_000

        uses_today += 1
        remaining_after = max(0, limit - uses_today)

        uref.set(
            {
                "user_id": int(user_id),
                "username": username or None,
                "display": display,
                "fish_inventory": fish_inv,
                "satoshi_note": int(note_cnt),
                "defense_tickets_list": tickets_list,
                "defense_tickets": len(tickets_list),
                "fishing_uses_date": uses_date,
                "fishing_uses_today": uses_today,
                "last_seen": dt,
                "last_active_date": today,
            },
            merge=True,
        )

    price_line = ""
    if loot_name in prices:
        price_line = f"오늘 시세: {int(prices.get(loot_name, 0))}$WHAT"
    elif loot_name == FISHING_SATOSHI_NOTE:
        price_line = "오늘 시세: 100000$WHAT"

    return {
        "ok": True,
        "loot_name": loot_name,
        "loot_value": int(loot_value),
        "price_line": price_line,
        "remaining": int(remaining_after),
        "limit": int(limit),
        "rod_level": int(rod_lvl),
        "sword_level": int(sword_lvl),
    }


async def fish_market_daily_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed = get_allowed_chat_id()
    if allowed is None:
        return
    db = get_firebase_client()
    dt = now_kst()
    await _ensure_daily_fish_prices(db, int(allowed), dt, force=True)
PALS_FEED_PRICE_EXP = 50

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
    "adult": "adult",
    "ultimate": "gg",
}

PALS_EVOLVE_AT: Dict[str, int] = {
    "baby": 500,
    "teen": 3_500,
    "adult": 10_000,
}

PALS_PAYOUT_EXP: Dict[str, int] = {
    "baby": 100,
    "teen": 1000,
    "adult": 2000,
    "ultimate": 5000,
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
BASED_MALL_PRICE_EXP = 70


SWORD_TABLE: Dict[int, Dict[str, Any]] = {
    0: {"name": "오래된 Based 나무 검", "cost": 0, "rate": 1.0, "sell": 5},
    1: {"name": "실버 Based 검", "cost": 35, "rate": 0.85, "sell": 80},
    2: {"name": "실버+ 검", "cost": 56, "rate": 0.80, "sell": 180},
    3: {"name": "골드 Based 검", "cost": 84, "rate": 0.75, "sell": 350},
    4: {"name": "골드+ 검", "cost": 126, "rate": 0.70, "sell": 650},
    5: {"name": "플래티넘 Based 검", "cost": 175, "rate": 0.65, "sell": 1100},
    6: {"name": "플래티넘+ 검", "cost": 245, "rate": 0.55, "sell": 1800},
    7: {"name": "루비 Based 검", "cost": 350, "rate": 0.48, "sell": 3000},
    8: {"name": "루비+ 검", "cost": 490, "rate": 0.42, "sell": 5000},
    9: {"name": "사파이어 Based 검", "cost": 700, "rate": 0.36, "sell": 8500},
    10: {"name": "사파이어+ 검", "cost": 980, "rate": 0.30, "sell": 15000},
    11: {"name": "오닉스 Based 검", "cost": 1400, "rate": 0.24, "sell": 26000},
    12: {"name": "오닉스+ 검", "cost": 1960, "rate": 0.19, "sell": 45000},
    13: {"name": "블러드 Based 검", "cost": 2660, "rate": 0.14, "sell": 80000},
    14: {"name": "블러드+ 검", "cost": 3640, "rate": 0.10, "sell": 150000},
    15: {"name": "검은 왕의 검", "cost": 4900, "rate": 0.065, "sell": 280000},
    16: {"name": "세계절단 검", "cost": 6300, "rate": 0.04, "sell": 500000},
    17: {"name": "신의 시험 검", "cost": 8400, "rate": 0.025, "sell": 900000},
    18: {"name": "멸망의 Based 검", "cost": 11200, "rate": 0.015, "sell": 1600000},
    19: {"name": "신화의 끝 검", "cost": 15400, "rate": 0.008, "sell": 3000000},
    20: {"name": "비탈릭 바짓속 불타는 명멸검", "cost": 21000, "rate": 0.002, "sell": None},
}


def sword_sell_price(lvl: int) -> Optional[int]:
    row = SWORD_TABLE.get(int(lvl))
    if not row:
        return None
    sell = row.get("sell")
    if sell is None:
        return None
    return int(sell)


DEFENSE_TICKET_TTL_SECONDS = 24 * 60 * 60


def _coerce_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    return None


def defense_tickets_list_from_udata(
    udata: Dict[str, Any], now: datetime
) -> Tuple[List[datetime], bool]:
    changed = False
    raw_list = udata.get("defense_tickets_list")
    tickets: List[datetime] = []
    if isinstance(raw_list, list):
        for v in raw_list:
            dtv = _coerce_dt(v)
            if dtv is not None:
                tickets.append(dtv)

    old_cnt = int(udata.get("defense_tickets", 0))
    if old_cnt > 0 and not tickets:
        exp = now + timedelta(seconds=DEFENSE_TICKET_TTL_SECONDS)
        tickets = [exp for _ in range(old_cnt)]
        changed = True

    before = len(tickets)
    tickets = [t for t in tickets if t > now]
    if len(tickets) != before:
        changed = True

    tickets.sort()
    if raw_list is not None and isinstance(raw_list, list):
        if len(raw_list) != len(tickets):
            changed = True

    return tickets, changed


def defense_tickets_count(udata: Dict[str, Any], now: datetime) -> int:
    tickets, _ = defense_tickets_list_from_udata(udata, now)
    return len(tickets)


def _format_remaining_hhmm(seconds: int) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h:02d}:{m:02d}"


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
    return


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

    docs = list(users_coll.where(filter=FieldFilter("username", "==", uname)).limit(1).stream())
    if not docs:
        docs = list(users_coll.where(filter=FieldFilter("username", "==", uname.lower())).limit(1).stream())

    if not docs:
        await update.message.reply_text(f"@{uname} 유저를 찾을 수 없습니다.")
        return

    dt = now_kst()
    today = kst_date_str(dt)

    udoc = docs[0]
    udoc.reference.set(
        {
            "total_exp": 0,
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
            "익명 관리자 모드를 끄고 다시 `!지갑`을 입력해 주세요."
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
    await update.message.reply_text(
        f"{display}\n"
        f"현재 잔고: {total_exp}$WHAT"
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
    if update.message is None or update.message.text is None:
        return

    if update.effective_chat.type == ChatType.PRIVATE:
        text = update.message.text
        if text.strip().startswith("!보물추가"):
            if not is_owner(update):
                await update.message.reply_text("권한이 없습니다.")
                return
            allowed = get_allowed_chat_id()
            if allowed is None:
                await update.message.reply_text("설정된 채팅이 없습니다.")
                return

            parts = text.strip().split()
            cmds = [p.strip() for p in parts[1:] if p.strip()]
            if not cmds:
                await update.message.reply_text("추가할 보물 명령어를 같이 입력해 주세요. 예) !보물추가 !사랑그리고평화")
                return

            for c in cmds:
                if not c.startswith("!") or " " in c or len(c) < 2:
                    await update.message.reply_text("보물 명령어는 공백 없이 !로 시작해야 합니다. 예) !사랑그리고평화")
                    return

            db = get_firebase_client()
            dt = now_kst()
            async with get_chat_lock(int(allowed)):
                cref = chat_ref(db, int(allowed))
                csnap = cref.get()
                cdata = csnap.to_dict() if csnap.exists else {}
                extra = cdata.get("extra_treasure_map")
                if not isinstance(extra, dict):
                    extra = {}
                extra2 = dict(extra)
                added = 0
                skipped = 0
                for c in cmds:
                    if c in extra2:
                        skipped += 1
                        continue
                    key = "extra_" + hashlib.md5(c.encode("utf-8")).hexdigest()[:12]
                    extra2[c] = key
                    added += 1
                cref.set(
                    {
                        "chat_id": int(allowed),
                        "extra_treasure_map": extra2,
                        "last_seen": dt,
                    },
                    merge=True,
                )
            await update.message.reply_text(f"보물 추가 완료: {added}개 (중복 스킵 {skipped}개)")
            return

        return

    if update.effective_chat.type not in (ChatType.SUPERGROUP, ChatType.GROUP):
        return

    if not is_allowed_chat(update):
        return

    text = update.message.text
    chat_id = int(update.effective_chat.id)

    if text.strip() in ("!알구매", "!먹이", "!마이팔"):
        await update.message.reply_text("해당 기능은 삭제되었습니다.")
        return

    if text.strip().startswith("!러시안룰렛"):
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 입력해 주세요."
            )
            return

        parts = text.strip().split()
        if len(parts) != 2:
            await update.message.reply_text("사용법: !러시안룰렛 @유저네임")
            return
        if not is_username_token(parts[1]):
            await update.message.reply_text("상대는 @유저네임 형태로 입력해주세요.")
            return

        chat_id_for_lock = int(update.effective_chat.id)
        async with get_rr_chat_lock(chat_id_for_lock):
            if get_active_rr(chat_id_for_lock) is not None:
                await update.message.reply_text("현재 진행 중인 러시안룰렛이 있습니다.")
                return

            challenger_id = int(update.effective_user.id)
            challenger_username = update.effective_user.username
            challenger_display = (
                f"@{challenger_username}"
                if challenger_username
                else (update.effective_user.full_name or str(challenger_id))
            )

            target_username = parse_username_token(parts[1])
            db = get_firebase_client()
            doc = _find_user_doc_by_username(db, chat_id_for_lock, target_username)
            if doc is None:
                await update.message.reply_text(f"@{target_username} 유저를 찾을 수 없습니다.")
                return
            try:
                opponent_id = int(doc.id)
            except Exception:
                await update.message.reply_text("대상 유저 정보가 유효하지 않습니다.")
                return
            if opponent_id == challenger_id:
                await update.message.reply_text("자기 자신에게는 러시안룰렛을 걸 수 없습니다.")
                return

            opponent_display = f"@{target_username}"

            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text="수락(300$WHAT)",
                            callback_data=f"rr_invite:{chat_id_for_lock}:{challenger_id}:{opponent_id}:yes",
                        ),
                        InlineKeyboardButton(
                            text="거절",
                            callback_data=f"rr_invite:{chat_id_for_lock}:{challenger_id}:{opponent_id}:no",
                        ),
                    ]
                ]
            )

            msg = await update.effective_chat.send_message(
                f"{opponent_display} 님 {challenger_display}님의 러시안룰렛을 승낙하시겠습니까?",
                reply_markup=kb,
            )

            game = {
                    "chat_id": chat_id_for_lock,
                    "message_id": int(msg.message_id),
                    "challenger_id": challenger_id,
                    "challenger_display": challenger_display,
                    "opponent_id": opponent_id,
                    "opponent_display": opponent_display,
                    "accepted": False,
                    "pot": 0,
                    "rps_choices": {},
                    "winner_id": None,
                    "turn_id": None,
                    "bullet_slot": random.randint(1, 6),
                    "picked_slots": [],
                    "phase": "invite",
                    "created_at": now_kst(),
                    "status_text": f"{opponent_display} 님 {challenger_display}님의 러시안룰렛을 승낙하시겠습니까?",
                    "last_reply_markup": kb,
                }
            set_active_rr(chat_id_for_lock, game)
            rr_start_invite_timeout(context, game)
        return

    if text.strip() == "!룰렛종료":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return
        chat_id_for_lock = int(update.effective_chat.id)
        async with get_rr_chat_lock(chat_id_for_lock):
            game = get_active_rr(chat_id_for_lock)
            if not game:
                await update.message.reply_text("진행 중인 러시안룰렛이 없습니다.")
                return
            rr_cancel_jobs(context, game)
            set_active_rr(chat_id_for_lock, None)
            mid = int(game.get("message_id") or 0)
        if mid:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id_for_lock,
                    message_id=mid,
                    text="방장에 의해 러시안룰렛이 종료되었습니다.",
                )
            except Exception:
                pass
        return

    treasure_map: Dict[str, str] = {}

    db = get_firebase_client()
    cref = chat_ref(db, chat_id)
    csnap = cref.get()
    cdata = csnap.to_dict() if csnap.exists else {}
    extra = cdata.get("extra_treasure_map")
    if isinstance(extra, dict):
        for cmd, key in extra.items():
            if not isinstance(cmd, str) or not cmd.strip().startswith("!"):
                continue
            if not isinstance(key, str) or not key.strip():
                continue
            treasure_map[cmd.strip()] = key.strip()

    if text.strip() == "!남은보물":
        dt = now_kst()
        async with get_chat_lock(chat_id):
            csnap = cref.get()
            cdata = csnap.to_dict() if csnap.exists else {}
            found = cdata.get("treasures_found_global")
            if not isinstance(found, dict):
                found = {}
            total_cnt = len(set(treasure_map.values()))
            found_cnt = 0
            for k in set(treasure_map.values()):
                if found.get(k):
                    found_cnt += 1
            remaining = max(0, total_cnt - found_cnt)
            cref.set(
                {
                    "chat_id": chat_id,
                    "last_seen": dt,
                },
                merge=True,
            )
        await update.message.reply_text(f"아직 숨겨져있는 보물은 총 {remaining}개 입니다.")
        return

    if text.strip() == "!보물해금":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return

        allowed = get_allowed_chat_id()
        if allowed is not None and int(allowed) != int(chat_id):
            return

        db = get_firebase_client()
        dt = now_kst()
        changed = await _refresh_daily_treasures(db, int(chat_id), dt, force=True)
        if not changed:
            await update.message.reply_text("이미 오늘의 보물이 준비되어 있습니다.")
            return
        await update.message.reply_text("보물 5개를 해금했습니다.")
        return

    if text.strip() == "!보물힌트초기화":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return

        allowed = get_allowed_chat_id()
        if allowed is not None and int(allowed) != int(chat_id):
            return

        db = get_firebase_client()
        dt = now_kst()
        today = kst_date_str(dt)

        users = list(chat_ref(db, int(chat_id)).collection("users").stream())
        cnt = 0
        for d in users:
            try:
                uid = int(d.id)
            except Exception:
                continue
            async with get_user_lock(int(chat_id), uid):
                uref = user_ref(db, int(chat_id), uid)
                uref.set(
                    {
                        "treasure_hint_date": today,
                        "treasure_hint_uses_today": 0,
                        "last_seen": dt,
                    },
                    merge=True,
                )
                cnt += 1

        await update.message.reply_text(f"보물힌트 횟수 초기화 완료 (대상 {cnt}명)")
        return

    if text.strip() == "!보물힌트":
        user_id = int(update.effective_user.id)
        dt = now_kst()
        today = kst_date_str(dt)

        async with get_user_lock(chat_id, user_id):
            uref = user_ref(db, chat_id, user_id)
            snap0 = uref.get()
            udata0 = snap0.to_dict() if snap0.exists else {}

            hint_date = udata0.get("treasure_hint_date")
            hint_uses = int(udata0.get("treasure_hint_uses_today", 0))
            if hint_date != today:
                hint_date = today
                hint_uses = 0

            charge = 0
            if hint_uses >= 2:
                charge = 80
                total_exp0 = int(udata0.get("total_exp", 0))
                if total_exp0 < charge:
                    await update.message.reply_text(
                        f"잔고가 부족합니다. (필요 {charge}$WHAT, 보유 {total_exp0}$WHAT)"
                    )
                    return
                uref.set({"total_exp": total_exp0 - charge}, merge=True)

            hint_uses += 1

            uref.set(
                {
                    "last_seen": dt,
                    "last_active_date": today,
                    "treasure_hint_date": hint_date,
                    "treasure_hint_uses_today": hint_uses,
                },
                merge=True,
            )

        async with get_chat_lock(chat_id):
            cref = chat_ref(db, chat_id)
            csnap = cref.get()
            cdata = csnap.to_dict() if csnap.exists else {}
            found = cdata.get("treasures_found_global")
            if not isinstance(found, dict):
                found = {}
            remaining_cmds: List[str] = []
            for cmd, key in treasure_map.items():
                if not found.get(key):
                    remaining_cmds.append(cmd)
            cref.set(
                {
                    "chat_id": chat_id,
                    "last_seen": dt,
                },
                merge=True,
            )

        if not remaining_cmds:
            await update.message.reply_text("남은 보물이 없습니다.")
            return
        pick = random.choice(remaining_cmds)
        hint = mask_treasure_hint(pick)
        suffix = ""
        if hint_uses > 2:
            suffix = " (80$WHAT 차감)"
        await update.message.reply_text(f"남은 보물의 명령어는 {hint} 입니다.{suffix}")
        return
    tkey = treasure_map.get(text.strip())
    if tkey is not None:
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 입력해 주세요."
            )
            return
        user_id = int(update.effective_user.id)
        dt = now_kst()
        today = kst_date_str(dt)

        async with get_chat_lock(chat_id):
            cref = chat_ref(db, chat_id)
            csnap = cref.get()
            cdata = csnap.to_dict() if csnap.exists else {}
            found = cdata.get("treasures_found_global")
            if not isinstance(found, dict):
                found = {}
            if found.get(tkey):
                await update.message.reply_text(
                    "해당 보물은 이미 비열한 파수꾼이 찾아감 ㄹㅇㅋㅋ 아쉽ㄲㅂㄲㅂ 스트레스 받을거야"
                )
                return
            found2 = dict(found)
            found2[tkey] = True
            cref.set(
                {
                    "chat_id": chat_id,
                    "treasures_found_global": found2,
                    "last_seen": dt,
                },
                merge=True,
            )

        async with get_user_lock(chat_id, user_id):
            uref = user_ref(db, chat_id, user_id)
            usnap = uref.get()
            udata = usnap.to_dict() if usnap.exists else {}
            total_exp = int(udata.get("total_exp", 0)) + TREASURE_REWARD_EXP
            uref.set(
                {
                    "total_exp": total_exp,
                    "last_seen": dt,
                    "last_active_date": today,
                },
                merge=True,
            )

        await update.message.reply_text("숨은 보물찾기에 성공하였습니다.")
        return

    if text.strip() == "!존스미스불러":
        await update.message.reply_text("@smithjohnyeah")
        return

    if text.strip() == "!존스미스":
        await update.message.reply_text(
            random.choice(
                [
                    "그만불러",
                    "부르지마",
                    "아임낫유얼파더",
                    "존스캠스",
                    "존스미싱",
                    "존스와핑",
                    "존스팽킹",
                    "존스미시",
                    "존스트레스",
                    "존스미마셍",
                    "존스트라이크",
                    "존스머프",
                    "존이스피싱",
                    "존스웨디시",
                    "존스크럽",
                    "존스근허다",
                    "존스미노프",
                    "존스미스포츈",
                    "존스미스트롯",
                    "존스미스미스미스",
                ]
            )
        )
        return

    if text.strip() == "/vc":
        await update.message.reply_text("댄스남한테 물어보세요")
        return

    if text.strip() == "!가이드":
        await update.message.reply_text(
            "존스미스 BOT 명령어 가이드\n"
            "\n"
            "[지갑]\n"
            "- !지갑: 내 잔고 확인\n"
            "\n"
            "[출석]\n"
            "- !출첵: 하루 1회 100$WHAT\n"
            "\n"
            "[메뉴 추천]\n"
            "- !점메추: 점심 메뉴 랜덤 추천\n"
            "- !저메추: 저녁 메뉴 랜덤 추천\n"
            "\n"
            "[덤벼고래 (가위바위보)]\n"
            "- !덤벼고래: 방장에게만 도전 가능한 가위바위보\n"
            "  (하루 2회, 이기면 방장 $WHAT에서 최대 50$WHAT 획득)\n"
            "\n"
            "[러시안룰렛]\n"
            "- !러시안룰렛 @상대: 러시안룰렛 도전(상대 수락 시 300$WHAT)\n"
            "- !러시안룰: 러시안룰렛 규칙/타임아웃 설명\n"
            "- !룰렛종료: 방장 전용 강제 종료\n"
            "\n"
            "[보물]\n"
            "- !남은보물: 남은 보물 개수 확인\n"
            "- !보물힌트: 남은 보물 중 랜덤 힌트\n"
            "\n"
            "[검 키우기]\n"
            "- !인벤토리: 현재 검/방어티켓 확인\n"
            "- !강화확률: 강화 단계별 비용/확률/판매가 확인\n"
            "- !오른: 강화 진행(확정 버튼)\n"
            "- !당근마켓: 검/물고기 판매(버튼)\n"
            f"- !베이스드몰: 검 구매({BASED_MALL_PRICE_EXP}$WHAT, 검이 없을 때만 가능)\n"
            "- !강화비용: 강화 비용/판매가\n"
            "\n"
            "[낚시]\n"
            "- !낚시: 낚시 1회 진행(하루 횟수 제한)\n"
            "- !낚시끝: 낚시 종료\n"
            "- !낚시법: 낚시 규칙/시세/판매/교환 설명\n"
            "- !월척확률: 낚시 드랍 확률표\n"
            "\n"
            "[기타]\n"
            "- !whoami: 내 USER_ID/USERNAME 확인\n"
        )
        return

    if text.strip() == "!낚시법":
        await update.message.reply_text(
            "[낚시 안내]\n"
            "- !낚시: 1회 낚시(캐스팅) 진행\n"
            "- !낚시끝: 낚시 종료(남은 횟수는 유지)\n"
            "\n"
            "[하루 낚시 횟수]\n"
            "- 기본: 10회\n"
            "- 장비 레벨 보정: 10 + (레벨*2)\n"
            "  (낚싯대 보유 시 낚싯대 레벨, 없으면 검 레벨 기준)\n"
            "\n"
            "[검↔낚싯대 교환]\n"
            "- !오른 에서 검→낚싯대 교환 가능\n"
            "- 낚싯대는 판매 불가(당근마켓에서 검 판매 안 됨)\n"
            "- 언제든 낚싯대→검으로 되돌려 강화/판매 가능\n"
            "\n"
            "[물고기 시세/판매]\n"
            "- 물고기 시세는 매일 00:00(KST)에 변동\n"
            "- !당근마켓에서 물고기 전부 판매 가능\n"
            "- 사토시의 비밀노트는 개당 100000$WHAT\n"
        )
        return

    if text.strip() == "!러시안룰":
        await update.message.reply_text(
            "[러시안룰렛 안내]\n"
            "- !러시안룰렛 @상대: 도전\n"
            "- 상대가 수락하면 300$WHAT이 걸립니다(수락자 300$WHAT 차감)\n"
            "\n"
            "[진행]\n"
            "1) 수락/거절\n"
            "2) 가위바위보로 승자 결정\n"
            "3) 승자가 선공/후공 선택\n"
            "4) 1~6 숫자 선택을 번갈아 진행(탄창 6칸, 총알 1발 랜덤)\n"
            "\n"
            "[승리/보상]\n"
            "- 총알 맞은 사람은 사망(패배)\n"
            "- 생존자가 300$WHAT 획득\n"
            "\n"
            "[타임아웃]\n"
            "- 초대 5분 내 미수락 시 자동 취소\n"
            "- 수락 이후 각 단계 30초 내 미선택 시 자동 기권패(상대가 300$WHAT 획득)\n"
            "\n"
            "[종료]\n"
            "- !룰렛종료: 방장 전용 강제 종료\n"
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
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}

            pal = udata.get("pal")
            if not isinstance(pal, dict):
                await update.message.reply_text("현재 Pals가 없습니다. 먼저 `!알구매`로 알을 구매해 주세요.")
                return

        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="50EXP",
                        callback_data=f"pals_feed:{chat_id}:{user_id}:50",
                    ),
                    InlineKeyboardButton(
                        text="100EXP",
                        callback_data=f"pals_feed:{chat_id}:{user_id}:100",
                    ),
                    InlineKeyboardButton(
                        text="500EXP",
                        callback_data=f"pals_feed:{chat_id}:{user_id}:500",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="먹이주기 종료",
                        callback_data=f"pals_feed_end:{chat_id}:{user_id}",
                    )
                ],
            ]
        )
        await update.message.reply_text(
            "몇 EXP를 소모하시겠습니까?\n"
            "(버튼은 `!먹이`를 입력한 본인만 누를 수 있습니다.)",
            reply_markup=kb,
        )
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
            f"{BASED_MALL_PRICE_EXP}$WHAT에 검을 팔고있습니다.",
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
            dt = now_kst()
            lvl, _ = sword_state_from_udata(udata)
            tickets_list, t_changed = defense_tickets_list_from_udata(udata, dt)
            tickets = len(tickets_list)
            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

            if t_changed:
                uref.set(
                    {
                        "defense_tickets_list": tickets_list,
                        "defense_tickets": tickets,
                        "last_seen": dt,
                    },
                    merge=True,
                )
            rod_lvl = udata.get("fishing_rod_level")
            fish_inv = _coerce_int_dict(udata.get("fish_inventory"))
            note_cnt = int(udata.get("satoshi_note", 0))

            prices = await _ensure_daily_fish_prices(db, chat_id, dt)

            if rod_lvl is not None and lvl == SWORD_NONE_LEVEL:
                rod_sell_price = sword_sell_price(int(rod_lvl))
                rod_sell_txt = "판매 불가" if rod_sell_price is None else f"{int(rod_sell_price)}$WHAT"
                lines = [
                    f"{display}님 현재 낚싯대 [{sword_name(int(rod_lvl))}]를 보유 중입니다.",
                    "- 판매: 불가 (낚싯대→검으로 교환 후 판매 가능)",
                    f"- 전환 시 검 판매가: {rod_sell_txt}",
                    f"강화 방어티켓: {tickets}장",
                ]
            elif lvl == SWORD_NONE_LEVEL:
                lines = [f"{display}님 현재 검이 없습니다.", f"강화 방어티켓: {tickets}장"]
            else:
                sell_price = sword_sell_price(lvl)
                sell_txt = "판매 불가" if sell_price is None else f"{int(sell_price)}$WHAT"
                lines = [
                    f"{display}님 현재소유 검 [{sword_name(lvl)}]이 있습니다.",
                    f"- 판매가: {sell_txt}",
                    f"강화 방어티켓: {tickets}장",
                ]

            fish_total = 0
            fish_lines: List[str] = []
            for name, cnt in sorted(fish_inv.items()):
                if int(cnt) <= 0:
                    continue
                p = int(prices.get(name, 0))
                subtotal = p * int(cnt)
                fish_total += subtotal
                fish_lines.append(f"- {name} x{int(cnt)} (개당 {p}$WHAT / {subtotal}$WHAT)")
            if note_cnt > 0:
                subtotal = 100_000 * int(note_cnt)
                fish_total += subtotal
                fish_lines.append(f"- {FISHING_SATOSHI_NOTE} x{int(note_cnt)} (개당 100000$WHAT / {subtotal}$WHAT)")

            if fish_lines:
                lines.append("\n[물고기 인벤토리(오늘 시세)]")
                lines.extend(fish_lines)
                lines.append(f"- 총액(예상): {int(fish_total)}$WHAT")

            for i, exp_at in enumerate(tickets_list, start=1):
                remain = int((exp_at - dt).total_seconds())
                lines.append(f"방어권{i} : {_format_remaining_hhmm(remain)} 남음")

            await update.message.reply_text("\n".join(lines))
        return

    if text.strip() in ("!낚시", "!낚시끝"):
        if is_anonymous_admin_message(update):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!낚시`를 입력해 주세요."
            )
            return

        chat_id = int(update.effective_chat.id)
        user_id = int(update.effective_user.id)
        username = update.effective_user.username
        display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

        if text.strip() == "!낚시끝":
            set_fishing_active(chat_id, user_id, False)
            set_fishing_pending(chat_id, user_id, False)
            await update.message.reply_text(f"{display} 낚시를 종료했습니다.")
            return

        db = get_firebase_client()
        dt = now_kst()
        async with get_user_lock(chat_id, user_id):
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            rod_lvl = udata.get("fishing_rod_level")
            if rod_lvl is None:
                await update.message.reply_text(
                    f"{display} 낚시는 낚싯대가 있어야 가능합니다.\n"
                    "!오른 에서 검→낚싯대 교환을 먼저 해주세요."
                )
                return
            tool_lvl = int(rod_lvl)
            limit = fishing_daily_limit(tool_lvl)
            uses_date = udata.get("fishing_uses_date")
            uses_today = int(udata.get("fishing_uses_today", 0))
            today = kst_date_str(dt)
            if uses_date != today:
                uses_today = 0
                uses_date = today
            remaining = max(0, limit - uses_today)

        if remaining <= 0:
            await update.message.reply_text(f"{display}님 오늘 낚시 가능 횟수를 모두 사용했습니다. (하루 {limit}회)")
            return

        set_fishing_active(chat_id, user_id, True)
        set_fishing_pending(chat_id, user_id, False)

        msg0 = await update.message.reply_text(f"{display} 낚시중...")
        fish_cancel_jobs(context, chat_id, user_id, int(msg0.message_id))

        res = await _do_fishing_cast(db, chat_id, user_id, username, display, dt)
        if not bool(res.get("ok")):
            set_fishing_active(chat_id, user_id, False)
            try:
                await msg0.edit_text(str(res.get("msg") or "낚시에 실패했습니다."))
            except Exception:
                pass
            return

        remaining2 = int(res.get("remaining") or 0)
        limit2 = int(res.get("limit") or 0)
        loot_name = str(res.get("loot_name") or "")
        loot_value = int(res.get("loot_value") or 0)
        price_line = str(res.get("price_line") or "").strip()

        can_continue = remaining2 > 0
        if not can_continue:
            set_fishing_active(chat_id, user_id, False)

        text2 = (
            f"{display} 낚시!\n"
            f"획득: {loot_name}\n"
            f"가치: {loot_value}$WHAT\n"
            + (price_line + "\n" if price_line else "")
            + f"남은 횟수: {remaining2}/{limit2}"
        )
        try:
            await msg0.edit_text(
                text2,
                reply_markup=_fishing_kb(chat_id, user_id, int(msg0.message_id), can_continue=can_continue),
            )
        except Exception:
            pass
        return

    if text.strip() == "!강화확률":
        lines: List[str] = []
        for lvl in range(1, SWORD_MAX_LEVEL + 1):
            row = SWORD_TABLE.get(lvl)
            if not row:
                continue
            rate = float(row["rate"]) * 100
            lines.append(
                f"{lvl}강: {rate:.2f}% (성공시 {row['name']})"
            )
        await update.message.reply_text("\n".join(lines))
        return

    if text.strip() == "!월척확률":
        weights = [80.0, 40.0, 20.0, 1.0, 0.05]
        labels = [
            "쓰레기(가치 0)",
            "흔한생선(80~150$WHAT)",
            "희귀생선(500~800$WHAT)",
            "강화 방어권(5000~8000$WHAT)",
            f"{FISHING_SATOSHI_NOTE}(100000$WHAT)",
        ]
        total = float(sum(weights))
        lines = ["낚시 확률(가중치 기준, 정규화)"]
        for label, w in zip(labels, weights):
            pct = (float(w) / total) * 100.0 if total > 0 else 0.0
            lines.append(f"- {label}: {pct:.4f}%")
        await update.message.reply_text("\n".join(lines))
        return

    if text.strip() == "!강화비용":
        lines: List[str] = ["[검 강화비용/판매가]"]
        for lvl in range(0, SWORD_MAX_LEVEL + 1):
            row = SWORD_TABLE.get(lvl)
            if not row:
                continue
            name = str(row.get("name") or "")
            cost = int(row.get("cost") or 0)
            sell = row.get("sell")
            sell_txt = "판매 불가" if sell is None else f"{int(sell)}$WHAT"
            lines.append(f"{lvl}강 {name}")
            lines.append(f"- 강화비용: {cost}$WHAT")
            lines.append(f"- 판매가: {sell_txt}")
            lines.append("")
        if lines and lines[-1] == "":
            lines = lines[:-1]
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
            dt = now_kst()
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, _ = sword_state_from_udata(udata)
            price = sword_sell_price(lvl)
            rod_lvl = udata.get("fishing_rod_level")
            fish_inv = _coerce_int_dict(udata.get("fish_inventory"))
            note_cnt = int(udata.get("satoshi_note", 0))
            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

            has_sword_sell = lvl != SWORD_NONE_LEVEL and price is not None
            has_fish_sell = bool(fish_inv) or note_cnt > 0

            if not has_sword_sell and not has_fish_sell:
                if rod_lvl is not None and lvl == SWORD_NONE_LEVEL:
                    await update.message.reply_text(f"{display}님은 낚싯대 상태라 검 판매가 불가능합니다. (물고기도 없음)")
                else:
                    await update.message.reply_text(f"{display}님 판매할 물건이 없습니다.")
                return

            prices = await _ensure_daily_fish_prices(db, chat_id, dt)
            fish_total = 0
            fish_lines: List[str] = []
            for name, cnt in sorted(fish_inv.items()):
                if cnt <= 0:
                    continue
                p = int(prices.get(name, 0))
                fish_total += p * int(cnt)
                fish_lines.append(f"- {name} x{int(cnt)} (개당 {p}$WHAT)")
            if note_cnt > 0:
                fish_total += 100_000 * int(note_cnt)
                fish_lines.append(f"- {FISHING_SATOSHI_NOTE} x{int(note_cnt)} (개당 100000$WHAT)")

            rows: List[List[InlineKeyboardButton]] = []
            if has_sword_sell:
                rows.append(
                    [
                        InlineKeyboardButton(
                            text=f"검 판매 ({int(price)}$WHAT)",
                            callback_data=f"sword_sell:{chat_id}:{user_id}:yes",
                        )
                    ]
                )
            if has_fish_sell:
                rows.append(
                    [
                        InlineKeyboardButton(
                            text=f"물고기 전부 판매 (+{int(fish_total)}$WHAT)",
                            callback_data=f"fish_sell_all:{chat_id}:{user_id}:yes",
                        )
                    ]
                )
            rows.append(
                [
                    InlineKeyboardButton(
                        text="취소",
                        callback_data=f"market_close:{chat_id}:{user_id}:ok",
                    )
                ]
            )
            kb = InlineKeyboardMarkup(rows)

            lines = [f"{display} 당근마켓"]
            if has_sword_sell:
                lines.append(f"- 보유 검: [{sword_name(lvl)}] 판매가 {int(price)}$WHAT")
            if fish_lines:
                lines.append("- 보유 물고기/")
                lines.extend(fish_lines)
                lines.append(f"- 예상 판매 총액: {int(fish_total)}$WHAT")

            await update.message.reply_text("\n".join(lines), reply_markup=kb)
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
            dt = now_kst()
            uref = user_ref(db, chat_id, user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, _ = sword_state_from_udata(udata)
            tickets_list, t_changed = defense_tickets_list_from_udata(udata, dt)
            tickets = len(tickets_list)
            if t_changed:
                uref.set(
                    {
                        "defense_tickets_list": tickets_list,
                        "defense_tickets": tickets,
                        "last_seen": dt,
                    },
                    merge=True,
                )
            rod_lvl = udata.get("fishing_rod_level")
            nxt = sword_next_upgrade_info(lvl) if lvl != SWORD_NONE_LEVEL else None
            username = update.effective_user.username
            display = f"@{username}" if username else (update.effective_user.full_name or str(user_id))

            if lvl == SWORD_NONE_LEVEL and rod_lvl is None:
                await update.message.reply_text(f"{display}님 현재 검이 없습니다.")
                return

            nxt_level = 0
            rate = 0.0
            cost = 0
            sell = None
            nxt_name = ""
            sell_txt = "판매 불가"
            if nxt is not None:
                nxt_level, rate, cost, sell, nxt_name = nxt
                sell_txt = "판매 불가" if sell is None else f"{int(sell)}EXP"

            extra_lines: List[str] = []
            for i, exp_at in enumerate(tickets_list, start=1):
                remain = int((exp_at - dt).total_seconds())
                extra_lines.append(f"방어권{i} : {_format_remaining_hhmm(remain)} 남음")
            extra_txt = "\n" + "\n".join(extra_lines) if extra_lines else ""

            rows: List[List[InlineKeyboardButton]] = []
            if lvl != SWORD_NONE_LEVEL and nxt is not None:
                rows.append(
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
                )
            elif lvl != SWORD_NONE_LEVEL and nxt is None:
                rows.append(
                    [
                        InlineKeyboardButton(
                            text="취소",
                            callback_data=f"sword_enhance_stop:{chat_id}:{user_id}",
                        )
                    ]
                )

            if lvl != SWORD_NONE_LEVEL and rod_lvl is None:
                rows.append(
                    [
                        InlineKeyboardButton(
                            text="검→낚싯대 교환",
                            callback_data=f"rod_exchange:{chat_id}:{user_id}:to_rod",
                        )
                    ]
                )
            if rod_lvl is not None and lvl == SWORD_NONE_LEVEL:
                rows.append(
                    [
                        InlineKeyboardButton(
                            text="낚싯대→검 교환",
                            callback_data=f"rod_exchange:{chat_id}:{user_id}:to_sword",
                        )
                    ]
                )

            kb = InlineKeyboardMarkup(rows) if rows else None

            if lvl != SWORD_NONE_LEVEL:
                if nxt is None:
                    msg = f"{display}님은 이미 최종 검을 보유 중입니다.\n보유 방어티켓: {tickets}장" + extra_txt
                else:
                    msg = (
                        f"{display}님의 [{sword_name(lvl)}]을 강화 하시겠습니까?\n"
                        f"강화확률 {rate*100:.2f}%, 강화비용 {int(cost)}$WHAT\n"
                        f"강화 후 검[{nxt_name}] 당근마켓 시세 {sell_txt}\n"
                        f"보유 방어티켓: {tickets}장" + extra_txt
                    )
            else:
                msg = (
                    f"{display}님은 낚싯대를 보유 중입니다.\n"
                    f"낚싯대 등급: [{sword_name(int(rod_lvl))}]\n"
                    f"보유 방어티켓: {tickets}장" + extra_txt
                )

            await update.message.reply_text(msg, reply_markup=kb)
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

            uref.set(
                {
                    "user_id": user_id,
                    "username": username or None,
                    "display": display,
                    "total_exp": new_total,
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

    if text.strip() == "!덤벼리셋":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return
        chat_id_for_lock = int(update.effective_chat.id)
        async with get_yacha_chat_lock(chat_id_for_lock):
            dt_now = now_kst()
            today_kst = kst_date_str(dt_now)
            db = get_firebase_client()
            users_col = chat_ref(db, chat_id_for_lock).collection("users")
            targets = users_col.where(filter=FieldFilter("yacha_uses_date", "==", today_kst)).stream()
            cnt = 0
            for doc in targets:
                try:
                    users_col.document(str(doc.id)).set(
                        {
                            "yacha_uses_date": today_kst,
                            "yacha_uses_today": 0,
                            "last_seen": dt_now,
                        },
                        merge=True,
                    )
                    cnt += 1
                except Exception:
                    pass
        await update.message.reply_text(
            f"오늘 덤벼고래 사용 기록을 초기화했습니다. (대상 {cnt}명)"
        )
        return

    if text.strip() == "!횟수검거초기화":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return
        chat_id_for_lock = int(update.effective_chat.id)
        async with get_chat_lock(chat_id_for_lock):
            dt_now = now_kst()
            today_kst = kst_date_str(dt_now)
            db = get_firebase_client()
            users_col = chat_ref(db, chat_id_for_lock).collection("users")
            targets = users_col.where(
                filter=FieldFilter("mafia_catch_uses_date", "==", today_kst)
            ).stream()
            cnt = 0
            for doc in targets:
                try:
                    users_col.document(str(doc.id)).set(
                        {
                            "mafia_catch_uses_date": today_kst,
                            "mafia_catch_uses_today": 0,
                            "last_seen": dt_now,
                        },
                        merge=True,
                    )
                    cnt += 1
                except Exception:
                    pass
        await update.message.reply_text(
            f"오늘 검거 사용 기록을 초기화했습니다. (대상 {cnt}명)"
        )
        return

    if text.strip() == "!안덤벼":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return
        chat_id_for_lock = int(update.effective_chat.id)
        async with get_yacha_chat_lock(chat_id_for_lock):
            if get_active_duel(chat_id_for_lock) is None:
                await update.message.reply_text("현재 진행 중인 야차가 없습니다.")
                return
            set_active_duel(chat_id_for_lock, None)
            context.chat_data.pop("yacha_pending", None)
        await update.message.reply_text("진행 중인 야차를 모두 종료했습니다.")
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
            owner_tag = f"<a href=\"tg://user?id={int(owner_id)}\">방장</a>"
            await update.effective_chat.send_message(
                f"{owner_tag}님, 덤벼고래를 수락하시겠습니까?",
                parse_mode="HTML",
                reply_markup=kb,
            )
            return

    if text.strip() == "!비자금":
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
            await update.message.reply_text("얼마의 비자금을 바치시겠습니까?")
            return

        if step == "await_amount":
            t = text.strip()
            try:
                amount = int(t)
            except ValueError:
                await update.message.reply_text("비자금은 숫자로 입력해주세요.")
                return
            if amount <= 0:
                await update.message.reply_text("비자금은 1 이상의 숫자로 입력해주세요.")
                return

            chat_id = int(update.effective_chat.id)
            target_user_id = int(tribute.get("target_user_id") or 0)
            if target_user_id <= 0:
                context.chat_data.pop("tribute_mode", None)
                await update.message.reply_text("대상 유저 정보가 유효하지 않습니다. 다시 `!비자금`부터 진행해주세요.")
                return

            db = get_firebase_client()
            dt = now_kst()
            async with get_user_lock(chat_id, target_user_id):
                uref = user_ref(db, chat_id, target_user_id)
                snap = uref.get()
                udata = snap.to_dict() if snap.exists else {}

                prev_total = int(udata.get("total_exp", 0))
                new_total = prev_total + int(amount)

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
                        "last_seen": dt,
                    },
                    merge=True,
                )

            context.chat_data.pop("tribute_mode", None)
            owner_name = update.effective_user.full_name if update.effective_user else "방장"
            await update.effective_chat.send_message(
                f"{owner_name}님이 비열하게도 {target_display}님에게 {amount}$WHAT를 싸바싸바했습니다."
            )
            return

    if text.strip() == "!꿀꺽":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return
        context.chat_data["thanos_mode"] = True
        await update.message.reply_text("꿀꺽할 유저를 선택해주세요.")
        return

    if is_owner(update) and context.chat_data.get("thanos_mode"):
        t = text.strip()
        if t.startswith("!") and len(t) > 1 and " " not in t and t not in (
            "!지갑",
            "!reset_db",
            "!reset_db confirm",
            "!chat_id",
            "!whoami",
            "!꿀꺽",
        ):
            target_username = t[1:]
            context.chat_data["thanos_mode"] = False
            await reset_user_by_username(update, context, target_username)
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
        if text.strip().lower() in ("!지갑",):
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!지갑`을 입력해 주세요."
            )
        return

    if text.strip().lower() in ("!지갑",):
        await handle_exp_query(update, context)
        return

    if text.strip() == "!포상금":
        if not is_owner(update):
            await update.message.reply_text("권한이 없습니다.")
            return

        chat_id = int(update.effective_chat.id)
        owner_id = get_owner_user_id()
        if owner_id is None:
            await update.message.reply_text("OWNER_USER_ID 설정이 필요합니다.")
            return

        context.chat_data["bounty_mode"] = {"scope": "", "step": "select"}
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="특정유저(유저네임)",
                        callback_data=f"bounty_select:{chat_id}:{int(owner_id)}:user",
                    ),
                    InlineKeyboardButton(
                        text="모든유저",
                        callback_data=f"bounty_select:{chat_id}:{int(owner_id)}:all",
                    ),
                ]
            ]
        )
        await update.message.reply_text("포상금 지급 대상을 선택해주세요.", reply_markup=kb)
        return

    bounty = context.chat_data.get("bounty_mode")
    if is_owner(update) and bounty and isinstance(bounty, dict):
        step = str(bounty.get("step") or "")
        if step == "await_username":
            uname = _normalize_username(text.strip())
            if not uname:
                await update.message.reply_text("유저네임을 입력해주세요. (예: @username)")
                return
            bounty["username"] = uname
            bounty["step"] = "await_amount"
            context.chat_data["bounty_mode"] = bounty
            await update.message.reply_text("얼마를 지급하시겠습니까? (숫자)")
            return

        if step == "await_amount":
            t = text.strip()
            try:
                amount = int(t)
            except ValueError:
                await update.message.reply_text("금액은 숫자로 입력해주세요.")
                return
            if amount <= 0:
                await update.message.reply_text("금액은 1 이상의 숫자로 입력해주세요.")
                return

            chat_id = int(update.effective_chat.id)
            db = get_firebase_client()
            dt = now_kst()
            today = kst_date_str(dt)

            scope = str(bounty.get("scope") or "")
            if scope == "all":
                users = list(chat_ref(db, chat_id).collection("users").stream())
                cnt = 0
                for d in users:
                    try:
                        uid = int(d.id)
                    except Exception:
                        continue
                    async with get_user_lock(chat_id, uid):
                        uref = user_ref(db, chat_id, uid)
                        snap = uref.get()
                        udata = snap.to_dict() if snap.exists else {}
                        total = int(udata.get("total_exp", 0)) + int(amount)
                        uref.set(
                            {"total_exp": total, "last_seen": dt, "last_active_date": today},
                            merge=True,
                        )
                        cnt += 1
                context.chat_data.pop("bounty_mode", None)
                await update.effective_chat.send_message(
                    f"포상금 지급 완료! 총 {cnt}명에게 {amount}$WHAT 지급"
                )
                return

            if scope == "user":
                uname = str(bounty.get("username") or "")
                doc = _find_user_doc_by_username(db, chat_id, uname)
                if doc is None:
                    await update.message.reply_text(f"@{uname} 유저를 찾을 수 없습니다.")
                    return
                try:
                    target_user_id = int(doc.id)
                except Exception:
                    await update.message.reply_text("대상 유저 정보가 유효하지 않습니다.")
                    return

                async with get_user_lock(chat_id, target_user_id):
                    uref = user_ref(db, chat_id, target_user_id)
                    snap = uref.get()
                    udata = snap.to_dict() if snap.exists else {}
                    prev_total = int(udata.get("total_exp", 0))
                    new_total = prev_total + int(amount)
                    display = udata.get("display") or (f"@{udata.get('username')}" if udata.get("username") else str(target_user_id))
                    uref.set(
                        {"total_exp": new_total, "last_seen": dt, "last_active_date": today},
                        merge=True,
                    )

                context.chat_data.pop("bounty_mode", None)
                await update.effective_chat.send_message(
                    f"포상금 지급 완료! {display}님에게 {amount}$WHAT 지급"
                )
                return

            context.chat_data.pop("bounty_mode", None)
            await update.message.reply_text("포상금 설정이 유효하지 않습니다. 다시 `!포상금`부터 진행해주세요.")
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
        defense_target = cdata2.get("defense_target")
        if not isinstance(defense_target, int) or defense_target < 1:
            defense_target = random.randint(400, 600)
        edison_counter = int(cdata2.get("edison_counter", 0))
        bonus_exp = 0
        bonus_msg: List[str] = []
        counter += 1
        defense_counter += 1
        edison_counter += 1
        if counter >= 365:
            counter = 0
            bonus_exp += 100
            bonus_msg.append(
                "띠링! 존 스미스의 폐지줍기 발!!동!! 존스미스가 고사리손으로 폐지를 주워 어렵사리 마련한 돈을 성공적으로 빼앗았습니다. 100$WHAT 획득"
            )
        if edison_counter >= 777:
            edison_counter = 0
            bonus_exp += 500
            bonus_msg.append(
                f"존 스미스가 저점매집한 코인을 개미들에게 팔아 넘겼습니다. 바람잡이를 한 당신({display})에게 500$WHAT를 선사합니다."
            )
        if defense_counter >= int(defense_target):
            defense_counter = 0
            defense_target = random.randint(400, 600)
            lvl0, _ = sword_state_from_udata(udata)
            tickets_list, _ = defense_tickets_list_from_udata(udata, dt)
            tickets_list.append(dt + timedelta(seconds=DEFENSE_TICKET_TTL_SECONDS))
            tickets_list.sort()
            tickets0 = len(tickets_list)
            uref.set(
                {
                    "sword_level": lvl0,
                    "defense_tickets_list": tickets_list,
                    "defense_tickets": tickets0,
                },
                merge=True,
            )
            await update.effective_chat.send_message(
                "띠링! 강화 방어티켓을 한장 부여합니다."
            )
        easter_done = bool(cdata2.get("easter_bisd_done", False))
        easter_step = int(cdata2.get("easter_bisd_step", 0))
        easter_user_ids = cdata2.get("easter_bisd_user_ids")
        if not isinstance(easter_user_ids, list):
            easter_user_ids = []
        tmp_ids: List[int] = []
        for x in easter_user_ids:
            if isinstance(x, bool):
                continue
            if isinstance(x, int):
                tmp_ids.append(int(x))
                continue
            if isinstance(x, float) and x.is_integer():
                tmp_ids.append(int(x))
                continue
            if isinstance(x, str) and x.strip().isdigit():
                tmp_ids.append(int(x.strip()))
        easter_user_ids = tmp_ids

        easter_winners: Optional[List[int]] = None
        if not easter_done:
            expected = ["베", "이", "스", "드"]
            token = (text or "").strip()
            if token == expected[min(max(easter_step, 0), 3)] and len(token) == 1:
                if int(user_id) in easter_user_ids:
                    easter_step = 0
                    easter_user_ids = []
                else:
                    easter_user_ids = easter_user_ids + [int(user_id)]
                    easter_step += 1
                    if easter_step >= 4:
                        easter_done = True
                        easter_step = 4
                        easter_winners = list(easter_user_ids)
            else:
                easter_step = 0
                easter_user_ids = []

        if easter_winners is not None:
            context.chat_data["easter_bisd_winners"] = easter_winners

        cref.set(
            {
                "chat_id": chat_id,
                "title": chat_title,
                "last_seen": dt,
                "blessing_counter": counter,
                "defense_counter": defense_counter,
                "defense_target": defense_target,
                "edison_counter": edison_counter,
                "easter_bisd_done": easter_done,
                "easter_bisd_step": easter_step,
                "easter_bisd_user_ids": easter_user_ids,
            },
            merge=True,
        )

    if bonus_exp > 0:
        total0 = int(udata.get("total_exp", 0))
        total1 = total0 + int(bonus_exp)
        uref.set({"total_exp": total1}, merge=True)
        udata["total_exp"] = total1
        for m in bonus_msg:
            try:
                await update.effective_chat.send_message(m)
            except Exception:
                pass

    context.chat_data.pop("easter_bisd_winners", None)

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

    if can_count:
        exp_res = calculate_exp(text, dt)
        gained = exp_res.gained_exp
        if gained > 0:
            exp_events.append({"ts": dt, "exp": gained})
            total_exp += gained

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
            "exp_events": exp_events,
            "warn_count": warn_count,
            "warn_reset_at": warn_reset_at,
            "last_seen": dt,
            "last_active_date": today,
            "exp_gained_date": exp_gained_date,
            "exp_gained_today": exp_gained_today,
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

    return


TREASURE_REWARD_EXP = 300


TREASURE_DAILY_POOL: List[str] = [
    "!소리에아이구배가터져게빛나여거덕인지도몰르구여기에나우드라이크헤이러탑원포더척원더라이크스테이션동네사람들",
    "!맨정신이난젤싫어아무것도할수가없어",
    "!아무일도없었다",
    "!존스미스포츈의쌍권총난사",
    "!사람들은모두변하나봐",
    "!불꽃어리둥절원식",
    "!시작이제일무서워미룬이",
    "!존스미꾸라지한마리가온웅덩이를흐린다",
    "!암쏘쏘뤼벗알라뷰다거짓말이야몰랐어이제야알았어",
    "!피카츄라이츄파이리꼬츄",
    "!그란데사이즈로말입니다",
    "!베이스드는분명보여줄것이다",
    "!아임파인땡큐앤쥬",
    "!알러뷰3000",
    "!존스미스가게이란사실을알고있는가",
    "!이래도지랄저래도지랄",
    "!베이스드많이사랑해주세요",
    "!개리와기리리가두개그래서리쌍",
    "!디지털골드는없었다",
    "!아주입만열면그짓말이자동으로나와",
    "!이뭔개소리야",
    "!천사소녀네티",
    "!해리포터와인피니티워",
    "!거를타선이없다",
    "!검은머리외국인",
    "!상처를치료해줄사람어디없누",
    "!젠장또이상혁이야",
    "!엄그리고준투더식",
    "!헤어지던밤찬바람이불었다",
    "!왜또아픈상처에소금을뿌리십니까",
]


async def _refresh_daily_treasures(
    db: firestore.Client, chat_id: int, dt: datetime, force: bool = False
) -> bool:
    if not TREASURE_DAILY_POOL:
        return False

    today = kst_date_str(dt)
    async with get_chat_lock(int(chat_id)):
        cref = chat_ref(db, int(chat_id))
        csnap = cref.get()
        cdata = csnap.to_dict() if csnap.exists else {}

        if not force and cdata.get("treasure_daily_date") == today:
            extra0 = cdata.get("extra_treasure_map")
            if isinstance(extra0, dict) and len(extra0) == 5:
                return False

        idx = int(cdata.get("treasure_daily_pool_index", 0))
        pool_len = len(TREASURE_DAILY_POOL)
        idx = idx % pool_len

        picked: List[str] = []
        for i in range(5):
            picked.append(TREASURE_DAILY_POOL[(idx + i) % pool_len])

        extra2: Dict[str, str] = {}
        for cmd in picked:
            key = "daily_" + hashlib.md5(cmd.encode("utf-8")).hexdigest()[:12]
            extra2[cmd] = key

        try:
            cref.update(
                {
                    "extra_treasure_map": firestore.DELETE_FIELD,
                    "treasures_found_global": firestore.DELETE_FIELD,
                }
            )
        except Exception:
            pass

        cref.set(
            {
                "chat_id": int(chat_id),
                "extra_treasure_map": extra2,
                "treasures_found_global": {},
                "treasure_daily_pool_index": (idx + 5) % pool_len,
                "treasure_daily_date": today,
                "last_seen": dt,
            },
            merge=True,
        )

    return True


async def treasure_daily_add_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed = get_allowed_chat_id()
    if allowed is None:
        return
    if not TREASURE_DAILY_POOL:
        return

    db = get_firebase_client()
    dt = now_kst()

    chat_id = int(allowed)
    changed = await _refresh_daily_treasures(db, chat_id, dt)
    if not changed:
        return

    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "00시 신규 보물 5개가 추가되었습니다. "
                "보물은 각각 300$WHAT를 지급합니다"
            ),
        )
    except Exception:
        return


async def treasure_startup_ensure_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed = get_allowed_chat_id()
    if allowed is None:
        return
    db = get_firebase_client()
    dt = now_kst()
    await _refresh_daily_treasures(db, int(allowed), dt)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query is None:
        return

    q = update.callback_query
    data = (q.data or "").strip()
    try:
        await q.answer()
    except Exception:
        pass

    if data.startswith("pals_"):
        return

    if q.message is None or q.message.chat is None:
        return

    chat_id = int(q.message.chat.id)
    allowed = get_allowed_chat_id()
    if allowed is not None and int(allowed) != chat_id:
        return

    if data.startswith("bounty_select:"):
        parts = data.split(":")
        if len(parts) != 4:
            return
        _, cid, owner_id, scope = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(owner_id):
            try:
                await q.answer("권한이 없습니다.", show_alert=True)
            except Exception:
                return
            return
        if scope not in ("user", "all"):
            return

        if scope == "user":
            context.chat_data["bounty_mode"] = {"scope": "user", "step": "await_username"}
            try:
                await q.message.edit_text("포상금을 지급할 유저네임을 입력해주세요. (예: @username)")
            except Exception:
                pass
            return

        context.chat_data["bounty_mode"] = {"scope": "all", "step": "await_amount"}
        try:
            await q.message.edit_text("모든 유저에게 지급할 금액을 입력해주세요. (숫자)")
        except Exception:
            pass
        return

    if data.startswith("fish_cast:"):
        parts = data.split(":")
        if len(parts) != 4:
            return
        _, cid, uid, mid = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            try:
                await q.answer("본인만 누를 수 있습니다.", show_alert=True)
            except Exception:
                return
            return
        if q.message is None or int(q.message.message_id) != int(mid):
            return

        user_id = int(uid)
        if not is_fishing_active(chat_id, user_id):
            try:
                await q.answer("낚시가 종료되었습니다. 다시 !낚시로 시작하세요.", show_alert=True)
            except Exception:
                return
            return

        username = q.from_user.username if q.from_user else None
        display = f"@{username}" if username else ((q.from_user.full_name if q.from_user else "") or str(user_id))

        set_fishing_pending(chat_id, user_id, False)
        fish_cancel_jobs(context, chat_id, user_id, int(mid))

        db = get_firebase_client()
        dt = now_kst()
        res = await _do_fishing_cast(db, chat_id, user_id, username, display, dt)
        if not bool(res.get("ok")):
            set_fishing_active(chat_id, user_id, False)
            try:
                await q.message.edit_text(str(res.get("msg") or "낚시에 실패했습니다."))
            except Exception:
                pass
            return

        remaining = int(res.get("remaining") or 0)
        limit = int(res.get("limit") or 0)
        loot_name = str(res.get("loot_name") or "")
        loot_value = int(res.get("loot_value") or 0)
        price_line = str(res.get("price_line") or "").strip()

        can_continue = remaining > 0
        if not can_continue:
            set_fishing_active(chat_id, user_id, False)

        text = (
            f"{display} 낚시!\n"
            f"획득: {loot_name}\n"
            f"가치: {loot_value}$WHAT\n"
            + (price_line + "\n" if price_line else "")
            + f"남은 횟수: {remaining}/{limit}"
        )
        try:
            await q.message.edit_text(
                text,
                reply_markup=_fishing_kb(chat_id, user_id, int(mid), can_continue=can_continue),
            )
        except Exception:
            pass
        return

    if data.startswith("fish_end:"):
        parts = data.split(":")
        if len(parts) != 4:
            return
        _, cid, uid, mid = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            return
        if q.message is None or int(q.message.message_id) != int(mid):
            return

        user_id = int(uid)
        fish_cancel_jobs(context, chat_id, user_id, int(mid))
        set_fishing_active(chat_id, user_id, False)
        set_fishing_pending(chat_id, user_id, False)
        try:
            await q.message.edit_text("낚시를 종료했습니다.")
        except Exception:
            pass
        return

    if data.startswith("market_close:"):
        parts = data.split(":")
        if len(parts) != 4:
            return
        _, cid, uid, _ = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            return
        try:
            await q.message.edit_text("당근마켓을 종료했습니다.")
        except Exception:
            pass
        return

    if data.startswith("fish_sell_all:"):
        parts = data.split(":")
        if len(parts) != 4:
            return
        _, cid, uid, decision = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            try:
                await q.answer("본인만 누를 수 있습니다.", show_alert=True)
            except Exception:
                return
            return
        if decision != "yes":
            try:
                await q.message.edit_text("판매가 취소되었습니다.")
            except Exception:
                pass
            return

        db = get_firebase_client()
        dt = now_kst()
        today = kst_date_str(dt)
        target_user_id = int(uid)
        async with get_user_lock(chat_id, target_user_id):
            uref = user_ref(db, chat_id, target_user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            fish_inv = _coerce_int_dict(udata.get("fish_inventory"))
            note_cnt = int(udata.get("satoshi_note", 0))
            if not fish_inv and note_cnt <= 0:
                try:
                    await q.message.edit_text("판매할 물고기가 없습니다.")
                except Exception:
                    pass
                return

            prices = await _ensure_daily_fish_prices(db, chat_id, dt)
            total = 0
            for name, cnt in fish_inv.items():
                if int(cnt) <= 0:
                    continue
                total += int(prices.get(name, 0)) * int(cnt)
            if note_cnt > 0:
                total += 100_000 * int(note_cnt)

            prev_total = int(udata.get("total_exp", 0))
            new_total = prev_total + int(total)
            uref.set(
                {
                    "total_exp": new_total,
                    "fish_inventory": {},
                    "satoshi_note": 0,
                    "last_seen": dt,
                    "last_active_date": today,
                },
                merge=True,
            )

        try:
            await q.message.edit_text(f"물고기 판매 완료! +{int(total)}$WHAT")
        except Exception:
            pass
        return

    if data.startswith("rod_exchange:"):
        parts = data.split(":")
        if len(parts) != 4:
            return
        _, cid, uid, mode = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            try:
                await q.answer("본인만 누를 수 있습니다.", show_alert=True)
            except Exception:
                return
            return
        if mode not in ("to_rod", "to_sword"):
            return

        db = get_firebase_client()
        dt = now_kst()
        today = kst_date_str(dt)
        target_user_id = int(uid)
        async with get_user_lock(chat_id, target_user_id):
            uref = user_ref(db, chat_id, target_user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            sword_lvl, _ = sword_state_from_udata(udata)
            rod_lvl = udata.get("fishing_rod_level")

            if mode == "to_rod":
                if sword_lvl == SWORD_NONE_LEVEL:
                    try:
                        await q.answer("검이 없어 교환할 수 없습니다.", show_alert=True)
                    except Exception:
                        return
                    return
                if rod_lvl is not None:
                    try:
                        await q.answer("이미 낚싯대를 보유 중입니다.", show_alert=True)
                    except Exception:
                        return
                    return
                uref.set(
                    {
                        "sword_level": SWORD_NONE_LEVEL,
                        "fishing_rod_level": int(sword_lvl),
                        "last_seen": dt,
                        "last_active_date": today,
                    },
                    merge=True,
                )
                try:
                    await q.message.edit_text(
                        f"검을 낚싯대로 교환했습니다.\n"
                        f"현재 낚싯대: [{sword_name(int(sword_lvl))}]\n"
                        "언제든 !오른 에서 다시 검으로 교환할 수 있습니다."
                    )
                except Exception:
                    pass
                return

            if rod_lvl is None:
                try:
                    await q.answer("낚싯대가 없어 교환할 수 없습니다.", show_alert=True)
                except Exception:
                    return
                return
            if sword_lvl != SWORD_NONE_LEVEL:
                try:
                    await q.answer("이미 검을 보유 중입니다.", show_alert=True)
                except Exception:
                    return
                return

            uref.set(
                {
                    "sword_level": int(rod_lvl),
                    "fishing_rod_level": firestore.DELETE_FIELD,
                    "last_seen": dt,
                    "last_active_date": today,
                },
                merge=True,
            )
            try:
                await q.message.edit_text(
                    f"낚싯대를 검으로 교환했습니다.\n현재 검: [{sword_name(int(rod_lvl))}]"
                )
            except Exception:
                pass
        return

    if data.startswith("rr_invite:"):
        parts = data.split(":")
        if len(parts) != 5:
            return
        _, cid, challenger_id, opponent_id, decision = parts
        if int(cid) != chat_id:
            return

        # DEBUG: send trace to chat
        _dbg_parts = f"from={q.from_user.id if q.from_user else None} opp={opponent_id} dec={decision}"
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"[DEBUG] rr_invite: {_dbg_parts}")
        except Exception:
            pass

        if q.from_user is None or int(q.from_user.id) != int(opponent_id):
            try:
                await context.bot.send_message(chat_id=chat_id, text=f"[DEBUG] wrong user: from={q.from_user.id if q.from_user else None} != opp={opponent_id}")
            except Exception:
                pass
            return

        cur_mid = int(q.message.message_id)

        try:
            game = get_active_rr(chat_id)
            try:
                await context.bot.send_message(chat_id=chat_id, text=f"[DEBUG] game phase={game.get('phase') if game else 'None'}, game_opp={game.get('opponent_id') if game else 'N/A'}")
            except Exception:
                pass
            if game is None:
                await context.bot.send_message(chat_id=chat_id, text="유효하지 않은 초대입니다.")
                return
            if int(game.get("challenger_id")) != int(challenger_id) or int(game.get("opponent_id")) != int(opponent_id):
                await context.bot.send_message(chat_id=chat_id, text="유효하지 않은 초대입니다.")
                return
            if game.get("phase") != "invite":
                await context.bot.send_message(chat_id=chat_id, text="이미 진행 중이거나 종료된 초대입니다.")
                return

            if int(game.get("message_id") or 0) != cur_mid:
                game["message_id"] = cur_mid
                set_active_rr(chat_id, game)

            if decision != "yes":
                rr_cancel_jobs(context, game)
                set_active_rr(chat_id, None)
                await context.bot.send_message(chat_id=chat_id, text="러시안룰렛이 거절되었습니다.")
                return

            rr_cancel_jobs(context, game)

            db = get_firebase_client()
            dt = now_kst()
            uref = user_ref(db, chat_id, int(opponent_id))
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            bal = int(udata.get("total_exp", 0))
            if bal < 300:
                set_active_rr(chat_id, None)
                await context.bot.send_message(chat_id=chat_id, text="잔고가 부족하여 수락할 수 없습니다. (필요 300$WHAT)")
                return
            uref.set({"total_exp": bal - 300, "last_seen": dt}, merge=True)

            game["accepted"] = True
            game["pot"] = 300
            game["phase"] = "rps"
            game["rps_choices"] = {}
            set_active_rr(chat_id, game)

            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text="가위",
                            callback_data=f"rr_rps:{chat_id}:{challenger_id}:{opponent_id}:scissors",
                        ),
                        InlineKeyboardButton(
                            text="바위",
                            callback_data=f"rr_rps:{chat_id}:{challenger_id}:{opponent_id}:rock",
                        ),
                        InlineKeyboardButton(
                            text="보",
                            callback_data=f"rr_rps:{chat_id}:{challenger_id}:{opponent_id}:paper",
                        ),
                    ]
                ]
            )
            cdisp = user_link(int(challenger_id), str(game.get("challenger_display") or str(challenger_id)))
            odisp = user_link(int(opponent_id), str(game.get("opponent_display") or str(opponent_id)))
            base = (
                "존스미스의 러시안룰렛이 시작됩니다.\n"
                "탄창은 6칸이며 랜덤 칸에 총알이 장전되어 있습니다. 가위바위보를 하여 순서정하기를 시작합니다.\n"
                f"{cdisp}과 {odisp}는 가위 바위 보 중 하나를 골라주세요."
            )

            await rr_set_message(context, game, base, reply_markup=kb, countdown=30)
            set_active_rr(chat_id, game)
            rr_start_action_timeout(context, game, [int(challenger_id), int(opponent_id)])
        except Exception:
            try:
                set_active_rr(chat_id, None)
            except Exception:
                pass
            try:
                await context.bot.send_message(chat_id=chat_id, text="러시안룰렛 처리 중 오류가 발생했습니다.")
            except Exception:
                pass
        return

    if data.startswith("rr_rps:"):
        parts = data.split(":")
        if len(parts) != 5:
            return
        _, cid, challenger_id, opponent_id, choice = parts
        if int(cid) != chat_id:
            return
        if choice not in ("rock", "paper", "scissors"):
            return

        uid = int(q.from_user.id) if q.from_user else 0
        if uid not in (int(challenger_id), int(opponent_id)):
            return

        async with get_rr_chat_lock(chat_id):
            game = get_active_rr(chat_id)
            if game is None or game.get("phase") != "rps":
                return

            choices = game.get("rps_choices")
            if not isinstance(choices, dict):
                choices = {}
            choices[str(uid)] = choice
            game["rps_choices"] = choices
            set_active_rr(chat_id, game)

            if len(choices.keys()) < 2:
                base = "한 명이 선택했습니다. 상대의 선택을 기다리는 중..."
                await rr_set_message(context, game, base, reply_markup=game.get("last_reply_markup"), countdown=30)
                set_active_rr(chat_id, game)
                return

            a_id = int(challenger_id)
            b_id = int(opponent_id)
            a_choice = str(choices.get(str(a_id)))
            b_choice = str(choices.get(str(b_id)))
            res = rps_result(a_choice, b_choice)
            if res == 0:
                game["rps_choices"] = {}
                set_active_rr(chat_id, game)
                base = "가위바위보가 비겼습니다. 다시 선택해주세요."
                await rr_set_message(context, game, base, reply_markup=game.get("last_reply_markup"), countdown=30)
                set_active_rr(chat_id, game)
                return

            winner_id = a_id if res == 1 else b_id
            loser_id = b_id if winner_id == a_id else a_id
            game["winner_id"] = winner_id
            game["phase"] = "order"
            set_active_rr(chat_id, game)

            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text="선공",
                            callback_data=f"rr_order:{chat_id}:{winner_id}:{loser_id}:first",
                        ),
                        InlineKeyboardButton(
                            text="후공",
                            callback_data=f"rr_order:{chat_id}:{winner_id}:{loser_id}:second",
                        ),
                    ]
                ]
            )
            wdisp = user_link(
                winner_id,
                str(
                    game.get("challenger_display")
                    if winner_id == int(game.get("challenger_id"))
                    else game.get("opponent_display")
                ),
            )
            base = (
                f"가위바위보 결과 {wdisp} 님이 승리하였습니다.\n"
                "승리한 유저가 선공하시겠습니까 후공하시겠습니까?"
            )
            await rr_set_message(context, game, base, reply_markup=kb, countdown=30)
            set_active_rr(chat_id, game)
            rr_start_action_timeout(context, game, [int(winner_id)])
        return

    if data.startswith("rr_order:"):
        parts = data.split(":")
        if len(parts) != 5:
            return
        _, cid, winner_id, loser_id, order = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(winner_id):
            try:
                await q.answer("승리자만 선택할 수 있습니다.", show_alert=True)
            except Exception:
                return
            return
        if order not in ("first", "second"):
            return

        async with get_rr_chat_lock(chat_id):
            game = get_active_rr(chat_id)
            if game is None or game.get("phase") != "order":
                return
            if int(game.get("winner_id") or 0) != int(winner_id):
                return

            turn_id = int(winner_id) if order == "first" else int(loser_id)
            game["turn_id"] = turn_id
            game["phase"] = "pick"
            set_active_rr(chat_id, game)

            num_buttons = [
                InlineKeyboardButton(
                    text=str(i), callback_data=f"rr_pick:{chat_id}:{winner_id}:{loser_id}:{i}"
                )
                for i in range(1, 7)
            ]
            kb = InlineKeyboardMarkup([num_buttons[:3], num_buttons[3:]])
            tdisp = user_link(
                turn_id,
                str(
                    game.get("challenger_display")
                    if turn_id == int(game.get("challenger_id"))
                    else game.get("opponent_display")
                ),
            )
            base = f"{'선공' if order == 'first' else '후공'}하셨습니다. {tdisp}님이 먼저 숫자를 골라주세요."
            await rr_set_message(context, game, base, reply_markup=kb, countdown=30)
            set_active_rr(chat_id, game)
            rr_start_action_timeout(context, game, [int(turn_id)])
        return

    if data.startswith("rr_pick:"):
        parts = data.split(":")
        if len(parts) != 5:
            return
        _, cid, winner_id, loser_id, slot_s = parts
        if int(cid) != chat_id:
            return
        try:
            slot = int(slot_s)
        except Exception:
            return
        if slot < 1 or slot > 6:
            return

        async with get_rr_chat_lock(chat_id):
            game = get_active_rr(chat_id)
            if game is None or game.get("phase") != "pick":
                return

            turn_id = int(game.get("turn_id") or 0)
            if q.from_user is None or int(q.from_user.id) != turn_id:
                try:
                    await q.answer("지금 차례가 아닙니다.", show_alert=True)
                except Exception:
                    return
                return

            picked = list(game.get("picked_slots") or [])
            if slot in picked:
                try:
                    await q.answer("이미 선택된 숫자입니다.", show_alert=True)
                except Exception:
                    return
                return
            picked.append(slot)
            game["picked_slots"] = picked

            c_id = int(game.get("challenger_id"))
            o_id = int(game.get("opponent_id"))
            other_id = o_id if turn_id == c_id else c_id
            tdisp = user_link(
                turn_id,
                str(game.get("challenger_display") if turn_id == c_id else game.get("opponent_display")),
            )
            odisp = user_link(
                other_id,
                str(game.get("challenger_display") if other_id == c_id else game.get("opponent_display")),
            )

            if slot == int(game.get("bullet_slot")):
                survivor_id = other_id
                pot = int(game.get("pot") or 0)
                db = get_firebase_client()
                dt = now_kst()
                lock1, lock2 = await acquire_two_user_locks(chat_id, c_id, o_id)
                try:
                    sref = user_ref(db, chat_id, survivor_id)
                    ssnap = sref.get()
                    sudata = ssnap.to_dict() if ssnap.exists else {}
                    s_bal = int(sudata.get("total_exp", 0)) + pot
                    sref.set({"total_exp": s_bal, "last_seen": dt}, merge=True)
                finally:
                    release_two_user_locks(lock1, lock2)

                rr_cancel_jobs(context, game)
                set_active_rr(chat_id, None)
                sdisp = user_link(
                    survivor_id,
                    str(game.get("challenger_display") if survivor_id == c_id else game.get("opponent_display")),
                )
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"{tdisp}님이 {slot}번을 고르셨습니다. 딸깍 드르르륵~\n"
                            "총알을 발사합니다.\n"
                            f"탕~ {tdisp}님은 총알을 맞고 사망하셨습니다. {sdisp}님은 전리품으로 {pot}$WHAT을 획득하셨습니다."
                        ),
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
                return

            game["turn_id"] = other_id
            set_active_rr(chat_id, game)

            remaining = [i for i in range(1, 7) if i not in picked]
            num_buttons = [
                InlineKeyboardButton(
                    text=str(i), callback_data=f"rr_pick:{chat_id}:{winner_id}:{loser_id}:{i}"
                )
                for i in remaining
            ]
            rows: List[List[InlineKeyboardButton]] = []
            while num_buttons:
                rows.append(num_buttons[:3])
                num_buttons = num_buttons[3:]
            kb = InlineKeyboardMarkup(rows) if rows else None

            base = (
                f"{tdisp}님이 {slot}번을 고르셨습니다. 딸깍 드르르륵~\n"
                "총알을 발사합니다.\n"
                f"틱. {tdisp}님은 생존하셨습니다.\n\n"
                f"{odisp}님이 숫자를 골라주세요."
            )
            await rr_set_message(context, game, base, reply_markup=kb, countdown=30)
            set_active_rr(chat_id, game)
            rr_start_action_timeout(context, game, [int(other_id)])
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
                    oref.set({"total_exp": oexp2}, merge=True)
                    cref.set({"total_exp": cexp2}, merge=True)
                finally:
                    release_two_user_locks(lock1, lock2)

        set_active_duel(chat_id, None)
        transfer_line = (
            f"$WHAT 이체: {loser_display} → {winner_display} ({delta}$WHAT)"
            if delta > 0
            else "$WHAT 이체: 없음"
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
            dt = now_kst()
            lvl, _ = sword_state_from_udata(udata)
            tickets_list, t_changed = defense_tickets_list_from_udata(udata, dt)
            tickets = len(tickets_list)
            if lvl != SWORD_NONE_LEVEL:
                await q.message.edit_text("이미 검을 보유 중입니다.")
                return

            total_exp = int(udata.get("total_exp", 0))
            if total_exp < BASED_MALL_PRICE_EXP:
                await q.message.edit_text(f"$WHAT가 부족합니다. (필요 {BASED_MALL_PRICE_EXP}$WHAT)")
                return

            total_exp -= BASED_MALL_PRICE_EXP
            uref.set(
                {
                    "total_exp": total_exp,
                    "sword_level": BASED_MALL_SWORD_LEVEL,
                    "defense_tickets_list": tickets_list,
                    "defense_tickets": tickets,
                },
                merge=True,
            )

        await q.message.edit_text(
            f"구매 완료! [{sword_name(BASED_MALL_SWORD_LEVEL)}] 지급 완료. (-{BASED_MALL_PRICE_EXP}$WHAT)"
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
            dt = now_kst()
            lvl, _ = sword_state_from_udata(udata)
            tickets_list, t_changed = defense_tickets_list_from_udata(udata, dt)
            tickets = len(tickets_list)
            if lvl == SWORD_NONE_LEVEL:
                await q.message.edit_text("현재 검이 없습니다.")
                return
            price = sword_sell_price(lvl)
            if price is None:
                await q.message.edit_text("현재 검은 판매 불가입니다.")
                return
            prev_total = int(udata.get("total_exp", 0))
            new_total = prev_total + int(price)
            uref.set(
                {
                    "total_exp": new_total,
                    "sword_level": SWORD_NONE_LEVEL,
                    "defense_tickets_list": tickets_list,
                    "defense_tickets": tickets,
                },
                merge=True,
            )
        await q.message.edit_text(f"판매 완료! {int(price)}$WHAT를 획득했습니다.\n현재 검: 없음")
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
            dt = now_kst()
            lvl, _ = sword_state_from_udata(udata)
            tickets_list, t_changed = defense_tickets_list_from_udata(udata, dt)
            tickets = len(tickets_list)
            if t_changed:
                uref.set(
                    {
                        "defense_tickets_list": tickets_list,
                        "defense_tickets": tickets,
                        "last_seen": dt,
                    },
                    merge=True,
                )
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
                await q.message.edit_text(f"$WHAT가 부족합니다. (필요 {int(cost)}$WHAT)")
                return

            total_exp -= int(cost)
            success = random.random() < float(rate)
            if success:
                lvl2 = nxt_level
                msg = f"강화 성공! [{nxt_name}] 획득!"
            else:
                cashback = int(int(cost) * 0.30)
                total_exp += cashback
                cashback_msg = (
                    "대장장이 오른이 불쌍한 당신에게 Based 카드 캐시백 혜택을 줍니다 "
                    f"받은 캐시백 : {cashback}$WHAT"
                )
                if tickets > 0:
                    tickets_list = tickets_list[1:]
                    tickets = len(tickets_list)
                    lvl2 = lvl
                    msg = "강화 실패! 방어티켓 1장을 사용하여 검이 복구되었습니다.\n" + cashback_msg
                else:
                    lvl2 = SWORD_NONE_LEVEL
                    msg = "강화 실패! 검이 파괴되어 사라졌습니다.\n" + cashback_msg

            uref.set(
                {
                    "total_exp": total_exp,
                    "sword_level": lvl2,
                    "defense_tickets_list": tickets_list,
                    "defense_tickets": tickets,
                },
                merge=True,
            )

        if lvl2 == SWORD_NONE_LEVEL:
            kb2 = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text=f"나무 검 사기 ({BASED_MALL_PRICE_EXP}$WHAT)",
                            callback_data=f"sword_buy_wood:{chat_id}:{uid}",
                        ),
                        InlineKeyboardButton(
                            text="나가기",
                            callback_data=f"sword_enhance_stop:{chat_id}:{uid}",
                        ),
                    ]
                ]
            )
            await q.message.edit_text(
                f"{msg}\n남은 방어티켓: {tickets}장",
                reply_markup=kb2,
            )
        else:
            can_continue = sword_next_upgrade_info(lvl2) is not None
            row: List[InlineKeyboardButton] = []
            if can_continue:
                row.append(
                    InlineKeyboardButton(
                        text="한번 더 강화",
                        callback_data=f"sword_enhance:{chat_id}:{uid}:yes",
                    )
                )
            row.append(
                InlineKeyboardButton(
                    text="판매하기",
                    callback_data=f"sword_sell_prompt:{chat_id}:{uid}",
                )
            )
            row.append(
                InlineKeyboardButton(
                    text="취소",
                    callback_data=f"sword_enhance_stop:{chat_id}:{uid}",
                )
            )
            kb2 = InlineKeyboardMarkup([row])
            await q.message.edit_text(
                f"{msg}\n현재 검: [{sword_name(lvl2)}]\n남은 방어티켓: {tickets}장",
                reply_markup=kb2,
            )
        return

    if data.startswith("sword_sell_prompt:"):
        parts = data.split(":")
        if len(parts) != 3:
            return
        _, cid, uid = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            try:
                await q.answer("명령어를 친 본인만 누를 수 있습니다.", show_alert=True)
            except Exception:
                return
            return

        db = get_firebase_client()
        target_user_id = int(uid)
        async with get_user_lock(chat_id, target_user_id):
            uref = user_ref(db, chat_id, target_user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            lvl, _ = sword_state_from_udata(udata)
            if lvl == SWORD_NONE_LEVEL:
                await q.message.edit_text("현재 검이 없습니다.")
                return
            price = sword_sell_price(lvl)
            if price is None:
                await q.message.edit_text("현재 검은 판매 불가입니다.")
                return

        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="판매하기",
                        callback_data=f"sword_sell:{chat_id}:{uid}:yes",
                    ),
                    InlineKeyboardButton(
                        text="취소",
                        callback_data=f"sword_enhance_stop:{chat_id}:{uid}",
                    ),
                ]
            ]
        )
        await q.message.edit_text(
            f"현재 소유한 [{sword_name(lvl)}]을 파시겠습니까? 판매가격 {int(price)}$WHAT",
            reply_markup=kb,
        )
        return

    if data.startswith("sword_buy_wood:"):
        parts = data.split(":")
        if len(parts) != 3:
            return
        _, cid, uid = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            try:
                await q.answer("명령어를 친 본인만 누를 수 있습니다.", show_alert=True)
            except Exception:
                return
            return

        db = get_firebase_client()
        target_user_id = int(uid)
        async with get_user_lock(chat_id, target_user_id):
            uref = user_ref(db, chat_id, target_user_id)
            snap = uref.get()
            udata = snap.to_dict() if snap.exists else {}
            dt = now_kst()
            lvl, _ = sword_state_from_udata(udata)
            tickets_list, _ = defense_tickets_list_from_udata(udata, dt)
            tickets = len(tickets_list)
            if lvl != SWORD_NONE_LEVEL:
                await q.message.edit_text("이미 검을 보유 중입니다.")
                return

            total_exp = int(udata.get("total_exp", 0))
            if total_exp < BASED_MALL_PRICE_EXP:
                await q.message.edit_text(f"EXP가 부족합니다. (필요 {BASED_MALL_PRICE_EXP}EXP)")
                return

            total_exp -= BASED_MALL_PRICE_EXP
            uref.set(
                {
                    "total_exp": total_exp,
                    "sword_level": BASED_MALL_SWORD_LEVEL,
                    "defense_tickets_list": tickets_list,
                    "defense_tickets": tickets,
                },
                merge=True,
            )

        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="강화하기",
                        callback_data=f"sword_enhance:{chat_id}:{uid}:yes",
                    ),
                    InlineKeyboardButton(
                        text="취소",
                        callback_data=f"sword_enhance_stop:{chat_id}:{uid}",
                    ),
                ]
            ]
        )
        await q.message.edit_text(
            f"구매 완료! [{sword_name(BASED_MALL_SWORD_LEVEL)}] 지급 완료. (-{BASED_MALL_PRICE_EXP}EXP)",
            reply_markup=kb,
        )
        return

    if data.startswith("sword_enhance_stop:"):
        parts = data.split(":")
        if len(parts) != 3:
            return
        _, cid, uid = parts
        if int(cid) != chat_id:
            return
        if q.from_user is None or int(q.from_user.id) != int(uid):
            try:
                await q.answer("명령어를 친 본인만 누를 수 있습니다.", show_alert=True)
            except Exception:
                return
            return
        try:
            await q.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
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


MAFIA_STEAL_EXP = 150
MAFIA_CATCH_REWARD_EXP = 500
MAFIA_PER_CHAT = 2

EASTER_BISD_REWARD_EXP = 2000


def _display_from_udata_or_docid(udata: Dict[str, Any], doc_id: str) -> str:
    return str(
        udata.get("display")
        or (f"@{udata.get('username')}" if udata.get("username") else "")
        or doc_id
    )


def _is_recent_active(udata: Dict[str, Any], now: datetime) -> bool:
    last_seen = udata.get("last_seen")
    if not last_seen:
        return False
    try:
        return last_seen > now - timedelta(hours=48)
    except Exception:
        return False


def _mafia_alive_count(cdata: Dict[str, Any]) -> int:
    alive = cdata.get("mafia_alive_ids")
    if not isinstance(alive, list):
        return 0
    return len([x for x in alive if isinstance(x, int)])


def _select_mafia_ids(user_docs: List[Any], now: datetime) -> List[int]:
    candidates: List[int] = []
    for udoc in user_docs:
        udata = udoc.to_dict() or {}
        uid = int(udata.get("user_id", int(udoc.id)))
        if _is_recent_active(udata, now):
            candidates.append(uid)

    if len(candidates) < MAFIA_PER_CHAT:
        candidates = []
        for udoc in user_docs:
            udata = udoc.to_dict() or {}
            uid = int(udata.get("user_id", int(udoc.id)))
            candidates.append(uid)

    if len(set(candidates)) >= MAFIA_PER_CHAT:
        return random.sample(list(set(candidates)), MAFIA_PER_CHAT)
    return list(set(candidates))


async def mafia_ensure_initialized_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    db = get_firebase_client()
    dt = now_kst()
    today = kst_date_str(dt)

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
        mafia_cleared_date = cdata.get("mafia_cleared_date")
        if mafia_cleared_date == today:
            continue
        mafia_date = cdata.get("mafia_date")
        alive_ids = cdata.get("mafia_alive_ids")
        if not isinstance(alive_ids, list):
            alive_ids = []
        alive_cnt = len([x for x in alive_ids if isinstance(x, int)])
        if mafia_date == today and alive_cnt > 0:
            continue

        users = list(cdoc.reference.collection("users").stream())
        mafia_ids = _select_mafia_ids(users, dt)
        cdoc.reference.set(
            {
                "mafia_date": today,
                "mafia_alive_ids": mafia_ids,
                "mafia_all_ids": mafia_ids,
                "mafia_cleared_date": None,
                "last_seen": dt,
            },
            merge=True,
        )


async def mafia_rollover_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    db = get_firebase_client()
    dt = now_kst()
    today = kst_date_str(dt)

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
        mafia_ids = _select_mafia_ids(users, dt)

        cdoc.reference.set(
            {
                "mafia_date": today,
                "mafia_alive_ids": mafia_ids,
                "mafia_all_ids": mafia_ids,
                "mafia_cleared_date": None,
                "last_seen": dt,
            },
            merge=True,
        )

        alive_cnt = len(mafia_ids)
        try:
            await context.bot.send_message(
                chat_id=int(chat_id),
                text=(
                    "새로운 랜덤마피아 두명이 선정되었습니다."
                    "마피아는 잡히기 전까지 오전11시 오후3시 오후8시 랜덤유저의 EXP를 강탈합니다. "
                    f"현재 생존마피아 ({alive_cnt}/{MAFIA_PER_CHAT})"
                ),
            )
        except Exception:
            continue


async def mafia_night_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    db = get_firebase_client()
    dt = now_kst()
    today = kst_date_str(dt)

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

        alive_ids = cdata.get("mafia_alive_ids")
        if not isinstance(alive_ids, list):
            alive_ids = []
        alive_ids = [int(x) for x in alive_ids if isinstance(x, int)]
        alive_cnt = len(alive_ids)

        try:
            await context.bot.send_message(
                chat_id=int(chat_id),
                text=(
                    "마피아의 밤이 왔습니다. 마피아는 잡히기 전까지 오전11시 오후3시 오후8시에 각각 랜덤유저 한명의 EXP를 강탈합니다. "
                    f"현재 생존마피아 ({alive_cnt}/{MAFIA_PER_CHAT})"
                ),
            )
        except Exception:
            pass

        if alive_cnt <= 0:
            continue

        users = list(cdoc.reference.collection("users").stream())
        user_docs: Dict[int, Any] = {}
        for udoc in users:
            udata = udoc.to_dict() or {}
            uid = int(udata.get("user_id", int(udoc.id)))
            user_docs[uid] = udoc

        victim_pool: List[int] = []
        for uid, udoc in user_docs.items():
            if uid in alive_ids:
                continue
            udata = udoc.to_dict() or {}
            if int(udata.get("total_exp", 0)) <= 0:
                continue
            victim_pool.append(uid)

        if not victim_pool:
            continue

        for mafia_id in alive_ids:
            if not victim_pool:
                break
            victim_id = random.choice(victim_pool)
            victim_pool = [x for x in victim_pool if x != victim_id]

            mdoc = user_docs.get(mafia_id)
            vdoc = user_docs.get(victim_id)
            if mdoc is None or vdoc is None:
                continue

            mdata = mdoc.to_dict() or {}
            vdata = vdoc.to_dict() or {}

            vexp = int(vdata.get("total_exp", 0))
            steal = min(MAFIA_STEAL_EXP, max(0, vexp))
            if steal <= 0:
                continue

            victim_disp = _display_from_udata_or_docid(vdata, str(victim_id))
            try:
                await context.bot.send_message(
                    chat_id=int(chat_id),
                    text=(
                        f"마피아가 {victim_disp} 의 {steal}EXP를 주머니에 챙겼습니다. "
                        "리더보드가 나올때 사람들의 EXP를 잘 체크하여 추리해보세요."
                    ),
                )
            except Exception:
                pass

            v_new = vexp - steal
            mexp = int(mdata.get("total_exp", 0))
            m_new = mexp + steal

            vdoc.reference.set(
                {
                    "total_exp": v_new,
                    "current_level": compute_level(v_new)[0],
                    "last_seen": dt,
                    "last_active_date": today,
                },
                merge=True,
            )
            mdoc.reference.set(
                {
                    "total_exp": m_new,
                    "current_level": compute_level(m_new)[0],
                    "last_seen": dt,
                    "last_active_date": today,
                },
                merge=True,
            )


async def mafia_reveal_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    db = get_firebase_client()
    dt = now_kst()
    today = kst_date_str(dt)

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

        mafia_date = cdata.get("mafia_date")
        if mafia_date != today:
            continue

        mafia_ids = cdata.get("mafia_all_ids")
        if not isinstance(mafia_ids, list) or not mafia_ids:
            mafia_ids = cdata.get("mafia_alive_ids")
        if not isinstance(mafia_ids, list):
            mafia_ids = []
        mafia_ids = [int(x) for x in mafia_ids if isinstance(x, int)]
        if not mafia_ids:
            continue

        users = list(cdoc.reference.collection("users").stream())
        id_to_name: Dict[int, str] = {}
        for udoc in users:
            udata = udoc.to_dict() or {}
            uid = int(udata.get("user_id", int(udoc.id)))
            uname = udata.get("username")
            if isinstance(uname, str) and uname.strip():
                id_to_name[uid] = uname.strip()
                continue
            disp = udata.get("display")
            if isinstance(disp, str) and disp.strip():
                id_to_name[uid] = disp.strip().lstrip("@").strip()

        names: List[str] = []
        for mid in mafia_ids:
            names.append(id_to_name.get(mid, str(mid)))

        try:
            await context.bot.send_message(
                chat_id=int(chat_id),
                text=f"오늘의 마피아는 {', '.join(names)} 이었습니다.",
            )
        except Exception:
            continue


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
    application.job_queue.run_once(treasure_startup_ensure_job, when=0)
    application.job_queue.run_once(fish_market_daily_job, when=0)
    application.job_queue.run_daily(treasure_daily_add_job, time=time(0, 0, tzinfo=kst))
    application.job_queue.run_daily(fish_market_daily_job, time=time(0, 0, tzinfo=kst))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
