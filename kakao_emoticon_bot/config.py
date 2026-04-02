import logging
import os
import re
import sys

from dotenv import load_dotenv

# exe 실행 시 exe가 있는 폴더, 개발 시 프로젝트 루트
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(APP_DIR, ".env"))

EMOTICON_ID_REGEX = re.compile(r"https://e\.kakao\.com/t/.+")
SHARE_LINK_REGEX = re.compile(r"https://emoticon\.kakao\.com/items/.+")
LOG_LEVEL = logging.INFO
TELEGRAM_TOKEN = os.getenv("KAKAO_BOT_TOKEN")
KAKAO_AUTH_TOKEN = os.getenv("KAKAO_AUTH_TOKEN", "")
