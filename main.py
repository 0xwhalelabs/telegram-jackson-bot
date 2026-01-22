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


async def handle_exp_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_user is None:
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

    def txn_fn(txn: firestore.Transaction) -> Dict[str, Any]:
        snap = uref.get(transaction=txn)
        data = snap.to_dict() if snap.exists else {}

        qts: List[Dict[str, Any]] = list(data.get("exp_query_timestamps", []))
        cutoff = dt - timedelta(hours=24)
        qts = [x for x in qts if x.get("ts") and x["ts"] >= cutoff]

        if len(qts) >= 3:
            return {
                "ok": False,
                "msg": "24시간 내 !EXP 조회 횟수 제한(3회)에 도달했습니다.",
            }

        qts.append({"ts": dt})

        txn.set(
            uref,
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

        return {
            "ok": True,
            "level": level,
            "total_exp": total_exp,
            "remaining": remaining,
            "need": need,
            "progress": progress,
            "date": date_key,
        }

    txn = db.transaction()
    result = txn_fn(txn)

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

    if update.message is None or update.message.text is None:
        return

    if update.effective_chat.type not in (ChatType.SUPERGROUP, ChatType.GROUP):
        return

    text = update.message.text

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

    norm = normalize_text(text)

    contains_url = URL_PATTERN.search(text) is not None

    txn = db.transaction()

    def txn_fn(txn: firestore.Transaction) -> Dict[str, Any]:
        chat_snap = cref.get(transaction=txn)
        user_snap = uref.get(transaction=txn)

        cdata = chat_snap.to_dict() if chat_snap.exists else {}
        udata = user_snap.to_dict() if user_snap.exists else {}

        mute_until = udata.get("mute_until")
        if mute_until and mute_until > dt:
            return {"action": "muted"}

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

        last_messages: List[Dict[str, Any]] = list(udata.get("last_messages", []))
        last_messages = [
            m
            for m in last_messages
            if m.get("ts") and m["ts"] >= dt - timedelta(minutes=2)
        ]

        is_repeat = False
        for m in last_messages:
            prev_norm = m.get("norm") or ""
            if prev_norm == norm:
                is_repeat = True
                break
            if prev_norm and similarity(prev_norm, norm) >= 0.9:
                is_repeat = True
                break

        if contains_url or is_repeat:
            warn_count += 1
            warn_reset_at = dt + timedelta(hours=24)

            mute_info: Optional[Dict[str, Any]] = None
            if warn_count >= 3:
                minutes = next_mute_minutes(mute_tier_today)
                mute_tier_today += 1
                warn_count = 0
                mute_until_new = dt + timedelta(minutes=minutes)
                mute_info = {"minutes": minutes, "until": mute_until_new}
                txn.set(
                    uref,
                    {
                        "mute_until": mute_until_new,
                        "mute_tier_today": mute_tier_today,
                        "mute_tier_date": mute_tier_date,
                    },
                    merge=True,
                )

            txn.set(
                uref,
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

            last_messages.append({"raw": text, "norm": norm, "ts": dt})
            last_messages = sorted(last_messages, key=lambda x: x["ts"])[-8:]
            txn.set(uref, {"last_messages": last_messages}, merge=True)

            txn.set(
                cref,
                {
                    "chat_id": chat_id,
                    "title": chat_title,
                    "last_seen": dt,
                },
                merge=True,
            )

            return {
                "action": "blocked",
                "blocked_reason": "url" if contains_url else "repeat",
                "warn": warn_count,
                "muted": mute_info,
                "display": display,
            }

        exp_events: List[Dict[str, Any]] = list(udata.get("exp_events", []))
        exp_events = [
            e
            for e in exp_events
            if e.get("ts") and e["ts"] >= dt - timedelta(minutes=1)
        ]

        can_count = len(exp_events) < 3

        gained = 0
        levelup: Optional[Dict[str, Any]] = None

        total_exp = int(udata.get("total_exp", 0))
        prev_level = int(udata.get("current_level", compute_level(total_exp)[0]))

        if can_count:
            exp_res = calculate_exp(text, dt)
            gained = exp_res.gained_exp
            if gained > 0:
                exp_events.append({"ts": dt, "exp": gained})
                total_exp += gained

        new_level, progress, need = compute_level(total_exp)
        if new_level != prev_level:
            levelup = {"level": new_level}

        exp_gained_date = udata.get("exp_gained_date")
        exp_gained_today = int(udata.get("exp_gained_today", 0))
        if exp_gained_date != today:
            exp_gained_today = 0
            exp_gained_date = today

        if gained > 0:
            exp_gained_today += gained

        last_messages.append({"raw": text, "norm": norm, "ts": dt})
        last_messages = sorted(last_messages, key=lambda x: x["ts"])[-8:]

        txn.set(
            uref,
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
                "mute_until": None,
                "last_seen": dt,
                "last_active_date": today,
                "exp_gained_date": exp_gained_date,
                "exp_gained_today": exp_gained_today,
            },
            merge=True,
        )

        txn.set(
            cref,
            {
                "chat_id": chat_id,
                "title": chat_title,
                "last_seen": dt,
            },
            merge=True,
        )

        return {
            "action": "ok",
            "gained": gained,
            "levelup": levelup,
            "display": display,
            "fever": is_fever_time(dt),
        }

    result = txn_fn(txn)

    if result.get("action") == "blocked":
        await maybe_delete_message(update, context)

        warn = result.get("warn", 0)
        display = result.get("display", "")
        await update.effective_chat.send_message(
            f"⚠️ {display}님 도배/스팸 감지!\n경고 {warn}/3"
        )

        muted = result.get("muted")
        if muted:
            until = muted["until"]
            minutes = muted["minutes"]
            await restrict_user(context, chat_id, user_id, until)
            await update.effective_chat.send_message(
                f"🔇 {display}님 경고 누적으로 {minutes}분간 뮤트 처리되었습니다."
            )
        return

    if result.get("action") == "ok":
        levelup = result.get("levelup")
        if levelup:
            await update.effective_chat.send_message(
                f"🎉 {result['display']}님 레벨 업!\n현재 레벨 Lv.{levelup['level']}"
            )


async def send_fever_start(context: ContextTypes.DEFAULT_TYPE) -> None:
    db = get_firebase_client()
    dt = now_kst()

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


async def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

    application = Application.builder().token(token).build()

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    from zoneinfo import ZoneInfo

    kst = ZoneInfo(KST_TZ)
    application.job_queue.run_daily(send_fever_start, time=time(19, 0, tzinfo=kst))
    application.job_queue.run_daily(send_fever_end, time=time(23, 0, tzinfo=kst))
    application.job_queue.run_repeating(send_leaderboard, interval=3 * 60 * 60, first=10)

    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)

    await application.updater.idle()

    await application.stop()
    await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
