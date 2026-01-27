import asyncio
import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from firebase_admin import credentials, firestore
import firebase_admin
from telegram import ChatPermissions, InlineKeyboardButton, InlineKeyboardMarkup, Update
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

        qts: List[Dict[str, Any]] = list(data.get("exp_query_timestamps", []))
        cutoff = dt - timedelta(hours=24)
        qts = [x for x in qts if x.get("ts") and x["ts"] >= cutoff]

        if len(qts) >= 3:
            await update.message.reply_text("24시간 내 !EXP 조회 횟수 제한(3회)에 도달했습니다.")
            return

        qts.append({"ts": dt})
        uref.set(
            {
                "user_id": user_id,
                "username": username or None,
                "display": display,
                "exp_query_timestamps": qts,
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

    if text.strip() == "!야차뜨자" or is_username_token(text.strip()):
        chat_id_for_lock = int(update.effective_chat.id)
        async with get_yacha_chat_lock(chat_id_for_lock):
            dt_now = now_kst()

            pending0 = context.chat_data.get("yacha_pending")
            if pending0 and isinstance(pending0, dict):
                started_at0 = pending0.get("started_at")
                if started_at0 and started_at0 < dt_now - timedelta(minutes=10):
                    context.chat_data.pop("yacha_pending", None)

            duel0 = get_active_duel(chat_id_for_lock)
            if duel0 and isinstance(duel0, dict):
                created_at0 = duel0.get("created_at")
                accepted0 = bool(duel0.get("accepted"))
                timeout = timedelta(minutes=30) if accepted0 else timedelta(minutes=10)
                if created_at0 and created_at0 < dt_now - timeout:
                    set_active_duel(chat_id_for_lock, None)

            if text.strip() == "!야차뜨자":
                if context.chat_data.get("yacha_pending"):
                    await update.message.reply_text("이미 야차 상대 선택 중입니다. 상대 @username을 입력해주세요.")
                    return

                if get_active_duel(chat_id_for_lock) is not None:
                    await update.message.reply_text("현재 진행 중인 야차가 있습니다.")
                    return

                db = get_firebase_client()
                dt = dt_now
                uref = user_ref(db, chat_id_for_lock, int(update.effective_user.id))
                snap = uref.get()
                data = snap.to_dict() if snap.exists else {}
                yacha_uses_date = data.get("yacha_uses_date")
                yacha_uses_today = int(data.get("yacha_uses_today", 0))
                today_kst = kst_date_str(dt)
                if yacha_uses_date != today_kst:
                    yacha_uses_date = today_kst
                    yacha_uses_today = 0
                if yacha_uses_today >= 5:
                    await update.message.reply_text("야차는 하루 5번만 사용할 수 있습니다.")
                    return
                yacha_uses_today += 1
                uref.set({"yacha_uses_date": yacha_uses_date, "yacha_uses_today": yacha_uses_today, "last_seen": dt}, merge=True)

                context.chat_data["yacha_pending"] = {
                    "challenger_id": int(update.effective_user.id),
                    "started_at": dt,
                }
                await update.message.reply_text("야차를 뜨실 악당을 선택해주세요.")
                return

            pending = context.chat_data.get("yacha_pending")
            if pending and isinstance(pending, dict):
                if int(pending.get("challenger_id", 0)) == int(update.effective_user.id) and is_username_token(text.strip()):
                    target_username = parse_username_token(text.strip())
                    db = get_firebase_client()
                    users_coll = chat_ref(db, chat_id_for_lock).collection("users")
                    docs = list(users_coll.where("username", "==", target_username).limit(1).stream())
                    if not docs:
                        docs = list(users_coll.where("username", "==", target_username.lower()).limit(1).stream())
                    if not docs:
                        await update.message.reply_text(f"@{target_username} 유저를 찾을 수 없습니다.")
                        return

                    target_doc = docs[0]
                    target_data = target_doc.to_dict() or {}
                    opponent_id = int(target_data.get("user_id", int(target_doc.id)))
                    if opponent_id == int(update.effective_user.id):
                        await update.message.reply_text("본인은 상대가 될 수 없습니다.")
                        return

                    if get_active_duel(chat_id_for_lock) is not None:
                        await update.message.reply_text("현재 진행 중인 야차가 있습니다.")
                        return

                    duel = {
                        "chat_id": chat_id_for_lock,
                        "challenger_id": int(update.effective_user.id),
                        "challenger_display": f"@{update.effective_user.username}" if update.effective_user.username else str(update.effective_user.id),
                        "opponent_id": opponent_id,
                        "opponent_username": target_username,
                        "accepted": False,
                        "choices": {},
                        "created_at": now_kst(),
                    }
                    set_active_duel(chat_id_for_lock, duel)
                    context.chat_data.pop("yacha_pending", None)

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
                        f"@{target_username}님 야차를 수락하시겠습니까?",
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
        opponent_display = f"@{str(duel.get('opponent_username') or '')}" if duel.get("opponent_username") else str(b_id)
        winner_display = challenger_display if winner_id == a_id else opponent_display
        loser_display = opponent_display if winner_id == a_id else challenger_display

        db = get_firebase_client()
        lock1, lock2 = await acquire_two_user_locks(chat_id, winner_id, loser_id)
        try:
            wref = user_ref(db, chat_id, winner_id)
            lref = user_ref(db, chat_id, loser_id)
            wsnap = wref.get()
            lsnap = lref.get()
            wdata = wsnap.to_dict() if wsnap.exists else {}
            ldata = lsnap.to_dict() if lsnap.exists else {}
            wexp = int(wdata.get("total_exp", 0))
            lexp = int(ldata.get("total_exp", 0))
            delta = min(50, max(0, lexp))
            wexp2 = wexp + delta
            lexp2 = max(0, lexp - delta)
            wlevel2 = compute_level(wexp2)[0]
            llevel2 = compute_level(lexp2)[0]
            wref.set({"total_exp": wexp2, "current_level": wlevel2}, merge=True)
            lref.set({"total_exp": lexp2, "current_level": llevel2}, merge=True)
        finally:
            release_two_user_locks(lock1, lock2)

        set_active_duel(chat_id, None)
        await q.message.edit_text(
            f"결과: {winner_display} 승!\n"
            f"EXP 이체: {loser_display} → {winner_display} ({delta} EXP)"
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

        top3 = user_rows[:3]
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
 
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
