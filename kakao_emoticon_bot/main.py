from telegram.ext import ApplicationBuilder, CommandHandler

from kakao_emoticon_bot.config import TELEGRAM_TOKEN
from kakao_emoticon_bot.message import create_emoticon, search_emoticon, set_auth_token, get_auth_from_desktop, merge_emoticons, create_emoji
from kakao_emoticon_bot.util import setup_logger


def main():
    if not TELEGRAM_TOKEN:
        print("Error: KAKAO_BOT_TOKEN 환경변수가 설정되지 않았습니다.")
        print(".env 파일에 KAKAO_BOT_TOKEN=your_token 을 추가하세요.")
        return

    setup_logger()

    telegram_application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .read_timeout(600)
        .write_timeout(600)
        .pool_timeout(600)
        .connect_timeout(600)
        .build()
    )

    telegram_application.add_handler(
        CommandHandler("create", create_emoticon)
    )
    telegram_application.add_handler(
        CommandHandler("search", search_emoticon)
    )
    telegram_application.add_handler(
        CommandHandler("setauth", set_auth_token)
    )
    telegram_application.add_handler(
        CommandHandler("getauth", get_auth_from_desktop)
    )
    telegram_application.add_handler(
        CommandHandler("merge", merge_emoticons)
    )
    telegram_application.add_handler(
        CommandHandler("emoji", create_emoji)
    )

    print("카카오 이모티콘 → 텔레그램 스티커 봇이 시작되었습니다.")
    telegram_application.run_polling()


if __name__ == "__main__":
    main()
