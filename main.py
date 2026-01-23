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
from telegram import ChatPermissions, Update
from telegram.constants import ChatType
from telegram.ext import Application, ContextTypes, MessageHandler, filters


load_dotenv()


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
    if len(message_text) < 10:
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
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
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

    await update.message.reply_text(
        f"{display}\n"
        f"현재 레벨: Lv.{result['level']}\n"
        f"현재 EXP: {result['total_exp']}\n"
        f"다음 레벨까지 남은 EXP: {result['remaining']}"
    )


async def maybe_delete_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    try:
        await update.message.delete()
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
        if text.strip().lower() == "!exp":
            await update.message.reply_text(
                "익명 관리자 모드로 보낸 메시지라 유저 식별이 불가능합니다.\n"
                "익명 관리자 모드를 끄고 다시 `!EXP`를 입력해 주세요."
            )
        return

    if text.strip().lower() == "!exp":
        await handle_exp_query(update, context)
        return

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

    if (not is_owner(update)) and contains_url:
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

    last_messages.append({"raw": text, "norm": norm, "ts": dt})
    last_messages = sorted(last_messages, key=lambda x: x["ts"])[-8:]

    uref.set(
        {
            "user_id": user_id,
            "username": username or None,
            "display": display,
            "total_exp": total_exp,
            "current_level": new_level,
            "exp_events": exp_events,
            "last_messages": last_messages,
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
