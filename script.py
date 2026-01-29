from dotenv import load_dotenv
load_dotenv()
import json
import logging
import glob
import os
from telegram.helpers import mention_html
import random
import re
from io import BytesIO
from typing import List, Dict, Tuple
from datetime import time  # <<< добавлено
import pytz  # <<< добавлено
from PIL import Image, ImageDraw, ImageFont

from telegram import Update, Message
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ================== НАСТРОЙКИ ================== #

# ⚠️ НЕ ХРАНИ ТОКЕН В КОДЕ. ЗАДАЙ В ОКРУЖЕНИИ:
# export BOT_TOKEN="..."
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

# ID владельца, который может использовать функционал бота В ЛС
OWNER_ID = 7877092881

# ID канала, в который бот должен писать (/babble, /say, авто-бред/опросы)
CHANNEL_ID = -1003009758716  # <<< твой канал

# Файл, где храним корпус токенов (слова + знаки)
CORPUS_FILE = "corpus_words.json"

# Файл, где храним эмодзи, которые встречались в канале
EMOJI_FILE = "corpus_emojis.json"

# Вероятность, что бот сам ответит в канал бредом после нового поста
AUTO_POST_PROBABILITY = 0.25  # 25% случаев

# Вероятность, что бот сам пришлёт ОПРОС после нового поста
AUTO_POLL_PROBABILITY = 0.10  # 10% (ТОЛЬКО авто-опросы, не команды)

# Вероятность подмешать эмодзи в сообщения бота (бред / say / babble и т.д.)
EMOJI_APPEND_PROBABILITY = 0.35  # 35%

# Вероятность иногда прислать ТОЛЬКО эмодзи (после поста), если есть собранные
AUTO_EMOJI_ONLY_PROBABILITY = 0.06  # 6%

# Вероятность, что бред будет адресован какому-то рандомному админу
RANDOM_ADMIN_MENTION_PROBABILITY = 0.3  # 30% случаев

# Вероятность рандомно оскорбить админа
RANDOM_ADMIN_INSULT_PROBABILITY = 0.08  # 8%

# Базовый триггер для текста /start
MEME_TRIGGER = "сделай меме"

# Список триггеров, которые бот ловит в ответах ("сделай меме" и т.п.)
MEME_TRIGGERS = ["сделай меме", "создай меме", "бля", "нахуй", "завоз"]

# Имя TTF-шрифта с кириллицей (должен лежать рядом со script.py)
MEME_FONT_FILE = "meme_font.ttf"

PUNCT = ".,!?#^£"

# Часовой пояс Москвы
MOSCOW_TZ = pytz.timezone("Europe/Moscow")

# ---- GPT генерация опросов ----
ENABLE_GPT_POLLS = True
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")  # можно переопределить окружением
# OPENAI_API_KEY задай в окружении:
# export OPENAI_API_KEY="..."

# =============================================== #

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Корпус токенов (слова/пунктуация)
CORPUS_TOKENS: List[str] = []

# Марковская цепь 2-го порядка: (w1, w2) -> [w3, w3, ...]
MARKOV2: Dict[Tuple[str, str], List[str]] = {}

# Эмодзи, которые встретились в канале
EMOJI_POOL: List[str] = []


# --------- ВСПОМОГАТЕЛЬНОЕ --------- #

def tokenize(text: str) -> List[str]:
    """Разбиваем текст на токены: слова/числа и знаки пунктуации . , ! ?"""
    tokens = re.findall(r"\w+|[.,!?]", text, flags=re.UNICODE)
    return tokens


def load_corpus_from_file():
    """Загружаем корпус токенов и строим марковскую цепь 2-го порядка."""
    global CORPUS_TOKENS, MARKOV2

    if not os.path.exists(CORPUS_FILE):
        logger.info("Файл корпуса не найден, начинаем с пустого.")
        CORPUS_TOKENS = []
        MARKOV2 = {}
        return

    try:
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                CORPUS_TOKENS = [str(w) for w in data]
            else:
                CORPUS_TOKENS = []
    except Exception as e:
        logger.error(f"Не удалось загрузить корпус: {e}")
        CORPUS_TOKENS = []

    # Перестраиваем марковскую цепь 2-го порядка
    MARKOV2 = {}
    if len(CORPUS_TOKENS) >= 3:
        for i in range(len(CORPUS_TOKENS) - 2):
            key = (CORPUS_TOKENS[i], CORPUS_TOKENS[i + 1])
            nxt = CORPUS_TOKENS[i + 2]
            MARKOV2.setdefault(key, []).append(nxt)

    logger.info(f"Загружено токенов в корпусе: {len(CORPUS_TOKENS)}")
    logger.info(f"Размер марковской цепи (2-й порядок): {len(MARKOV2)}")


def save_corpus_to_file():
    """Сохраняем корпус токенов в файл."""
    try:
        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(CORPUS_TOKENS, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Не удалось сохранить корпус: {e}")


def load_emojis():
    global EMOJI_POOL
    if not os.path.exists(EMOJI_FILE):
        EMOJI_POOL = []
        return
    try:
        with open(EMOJI_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                seen = set()
                cleaned = []
                for x in data:
                    s = str(x)
                    if s and s not in seen:
                        seen.add(s)
                        cleaned.append(s)
                EMOJI_POOL = cleaned
            else:
                EMOJI_POOL = []
    except Exception as e:
        logger.error(f"Не удалось загрузить эмодзи: {e}")
        EMOJI_POOL = []


def save_emojis():
    try:
        with open(EMOJI_FILE, "w", encoding="utf-8") as f:
            json.dump(EMOJI_POOL, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Не удалось сохранить эмодзи: {e}")


# Простенький детектор эмодзи по диапазонам Unicode
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)


def extract_emojis_from_text(text: str) -> List[str]:
    if not text:
        return []
    chunks = _EMOJI_RE.findall(text)
    out: List[str] = []
    for ch in chunks:
        for c in ch:
            if c.strip():
                out.append(c)
    return out


def add_emojis_from_message(msg: Message):
    """Добавляем эмодзи из текста/подписи/стикера в пул."""
    global EMOJI_POOL
    found: List[str] = []

    if msg.text:
        found.extend(extract_emojis_from_text(msg.text))
    if msg.caption:
        found.extend(extract_emojis_from_text(msg.caption))

    try:
        if msg.sticker and msg.sticker.emoji:
            found.extend(extract_emojis_from_text(msg.sticker.emoji) or [msg.sticker.emoji])
    except Exception:
        pass

    if not found:
        return

    seen = set(EMOJI_POOL)
    changed = False
    for e in found:
        if e and e not in seen:
            EMOJI_POOL.append(e)
            seen.add(e)
            changed = True

    if changed:
        save_emojis()


def pick_random_emoji() -> str:
    if not EMOJI_POOL:
        return ""
    return random.choice(EMOJI_POOL)


def maybe_append_emoji(text: str) -> str:
    """Иногда подмешиваем эмодзи в конец текста."""
    if EMOJI_POOL and random.random() < EMOJI_APPEND_PROBABILITY:
        e = pick_random_emoji()
        if e:
            if random.random() < 0.25:
                e2 = pick_random_emoji()
                if e2:
                    return f"{text} {e}{e2}"
            return f"{text} {e}"
    return text


async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Проверяем, админ ли пользователь в текущем чате.
    В ЛС админом считается только OWNER_ID.
    В группах/канале — обычная проверка статуса.
    """
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return False

    if chat.type == "private":
        return user.id == OWNER_ID

    member = await context.bot.get_chat_member(chat.id, user.id)
    return member.status in ("administrator", "creator")


async def get_random_admin(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Возвращает случайного НЕ-бота-админа этого чата (или None, если нет)."""
    try:
        admins = await context.bot.get_chat_administrators(chat_id)
    except Exception as e:
        logger.error(f"Не удалось получить админов для чата {chat_id}: {e}")
        return None

    humans = [a.user for a in admins if not a.user.is_bot]
    if not humans:
        return None
    return random.choice(humans)


def update_markov_with_sequence(seq: List[str]):
    """Обновляем марковскую цепь новыми токенами подряд (2-й порядок)."""
    global MARKOV2
    if not seq:
        return

    prev1 = prev2 = None
    if len(CORPUS_TOKENS) >= 2:
        prev1, prev2 = CORPUS_TOKENS[-2], CORPUS_TOKENS[-1]
    elif len(CORPUS_TOKENS) == 1:
        prev1, prev2 = CORPUS_TOKENS[-1], seq[0]

    for w in seq:
        if prev1 is not None and prev2 is not None:
            MARKOV2.setdefault((prev1, prev2), []).append(w)
        prev1, prev2 = prev2, w


def add_tokens_from_message(msg: Message):
    """Добавляем токены из текста/подписи сообщения в корпус и обновляем марковскую цепь."""
    global CORPUS_TOKENS
    text_parts = []
    if msg.text:
        text_parts.append(msg.text)
    if msg.caption:
        text_parts.append(msg.caption)

    joined = " ".join(text_parts)
    if not joined:
        return

    tokens = tokenize(joined)
    if not tokens:
        return

    update_markov_with_sequence(tokens)
    CORPUS_TOKENS.extend(tokens)
    save_corpus_to_file()


def pick_start_pair() -> Tuple[str, str] | None:
    """Выбираем стартовую пару токенов, желательно не начинающуюся с пунктуации."""
    if len(CORPUS_TOKENS) < 2:
        return None
    for _ in range(50):
        i = random.randint(0, len(CORPUS_TOKENS) - 2)
        w1, w2 = CORPUS_TOKENS[i], CORPUS_TOKENS[i + 1]
        if w1 not in PUNCT:
            return w1, w2
    i = random.randint(0, len(CORPUS_TOKENS) - 2)
    return CORPUS_TOKENS[i], CORPUS_TOKENS[i + 1]


def tokens_to_text(tokens: List[str]) -> str:
    """Склеиваем токены обратно в текст с аккуратной пунктуацией."""
    result = ""
    last_was_punct = False
    for t in tokens:
        if t in PUNCT:
            if not result:
                continue
            if last_was_punct:
                continue
            result = result.rstrip() + t + " "
            last_was_punct = True
        else:
            result += t + " "
            last_was_punct = False

    text = result.strip()
    if not text:
        return ""
    if text[-1] not in PUNCT:
        text += random.choice(["...", "!", "?!"])
    return text


def make_babble_markov2(max_tokens: int = None) -> str:
    """Генерим текст по марковской цепи 2-го порядка (1–13 слов)."""
    if max_tokens is None:
        max_tokens = random.randint(1, 13)

    if len(CORPUS_TOKENS) < 3 or not MARKOV2:
        return "Пока мало данных для марковской магии. Напишите что-нибудь в канал."

    start_pair = pick_start_pair()
    if not start_pair:
        return "Не удалось выбрать стартовую пару токенов."

    w1, w2 = start_pair
    tokens = [w1, w2]

    while len(tokens) < max_tokens:
        key = (tokens[-2], tokens[-1])
        candidates = MARKOV2.get(key)
        if not candidates:
            break
        nxt = random.choice(candidates)
        if nxt in PUNCT and tokens[-1] in PUNCT:
            continue
        tokens.append(nxt)

    tokens = tokens[:max_tokens]
    return tokens_to_text(tokens)


# --------- ПОЛЛЫ --------- #

def _random_words(n_min: int, n_max: int) -> List[str]:
    """Берём случайные 'слова' из корпуса (без пунктуации)."""
    words = [t for t in CORPUS_TOKENS if t not in PUNCT and len(t) > 0]
    if not words:
        return []
    n = random.randint(n_min, n_max)
    return random.sample(words, k=min(n, len(words)))


def generate_random_poll() -> Tuple[str, List[str]]:
    """Генерируем: (вопрос, варианты) 2–5."""
    if len(CORPUS_TOKENS) >= 3 and MARKOV2 and random.random() < 0.7:
        q = make_babble_markov2(max_tokens=random.randint(3, 9))
    else:
        base = _random_words(2, 6)
        q = " ".join(base).strip()
        if not q:
            q = random.choice([
                "ну че как?",
                "кто сегодня красавчик?",
                "что выбираем?",
                "вопрос века:",
                "ну давай голосование",
            ])
        if q[-1] not in "?!":
            q += random.choice(["?", "?!"])

    q = q[:290]

    option_count = random.randint(2, 5)
    options_set = set()
    options: List[str] = []

    attempts = 0
    while len(options) < option_count and attempts < 200:
        attempts += 1
        parts = _random_words(1, 3)
        if not parts:
            candidate = random.choice(["да", "нет", "возможно", "смотря", "я пас"])
        else:
            candidate = " ".join(parts)

        candidate = candidate.strip()[:95]
        if not candidate:
            continue
        low = candidate.lower()
        if low in options_set:
            continue
        options_set.add(low)
        options.append(candidate)

    while len(options) < 2:
        fallback = random.choice(["да", "нет", "не знаю", "жесть", "кайф"])
        if fallback.lower() not in options_set:
            options.append(fallback)
            options_set.add(fallback.lower())

    return q, options


def maybe_append_emoji_to_option(opt: str) -> str:
    """Иногда добавляем эмодзи в вариант."""
    if not opt:
        return opt
    if EMOJI_POOL and random.random() < (EMOJI_APPEND_PROBABILITY * 0.6):
        e = pick_random_emoji()
        if e:
            return f"{opt} {e}" if random.random() < 0.7 else f"{opt}{e}"
    return opt


def parse_poll_payload(raw: str) -> Tuple[str, List[str], int | None] | None:
    """
    Парсит: вопрос | вариант1 | вариант2 | ...
    Для quiz можно пометить правильный вариант звёздочкой:
      вопрос | *правильный | неправильный | ...
    """
    if not raw:
        return None

    parts = [p.strip() for p in raw.split("|") if p.strip()]
    if len(parts) < 3:
        return None

    question = parts[0][:290]
    options_raw = parts[1:11]  # максимум 10 вариантов

    correct_index = None
    options: List[str] = []

    for p in options_raw:
        if p.startswith("*"):
            p2 = p[1:].strip()
            if p2:
                if correct_index is None:
                    correct_index = len(options)
                options.append(p2[:95])
        else:
            options.append(p[:95])

    if len(options) < 2:
        return None

    return question, options, correct_index


def parse_poll_flags_and_rest(args_text: str) -> Tuple[dict, str]:
    """
    Флаги:
      anon, multi, quiz, gpt
    Остальное — payload (вопрос | варианты) ИЛИ тема для gpt.
    """
    flags = {"anon": False, "multi": False, "quiz": False, "gpt": False}

    if not args_text:
        return flags, ""

    tokens = args_text.strip().split()
    rest_tokens = []
    for t in tokens:
        low = t.lower()
        if low in ("anon", "multi", "quiz", "gpt"):
            flags[low] = True
        else:
            rest_tokens.append(t)

    rest = " ".join(rest_tokens).strip()
    return flags, rest


def generate_gpt_poll(topic: str, quiz: bool = False) -> Tuple[str, List[str], int | None]:
    """Генерит poll через OpenAI. Если нет ключа — кидает исключение."""
    if not ENABLE_GPT_POLLS:
        raise RuntimeError("GPT polls выключены (ENABLE_GPT_POLLS=False).")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Нет OPENAI_API_KEY в окружении.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Не установлен пакет openai. Поставь: pip install openai. ({e})")

    client = OpenAI(api_key=api_key)

    topic = (topic or "").strip()
    if not topic:
        topic = "смешной абсурдный опрос для телеграм-канала на русском"

    want = "с правильным вариантом" if quiz else "без правильного варианта"
    prompt = f"""
Сгенерируй короткий телеграм-опрос на русском {want}.
Верни СТРОГО JSON без текста вокруг, формат:
{{
  "question": "...",
  "options": ["...", "...", "..."],
  "correct_index": 0
}}
Правила:
- question до 120 символов
- options: 2–5 вариантов, каждый до 50 символов
- Без токсичных оскорблений, без призывов к насилию
- Тема/вдохновение: {topic}
Если это не quiz, ставь correct_index = null
"""

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )
    text = (getattr(resp, "output_text", None) or "").strip()
    if not text:
        raise RuntimeError("Пустой ответ от GPT.")

    try:
        data = json.loads(text)
        q = str(data.get("question", "")).strip()[:290]
        opts = data.get("options", [])
        if not isinstance(opts, list):
            raise ValueError("options не список")
        options = [str(x).strip()[:95] for x in opts if str(x).strip()]
        if len(options) < 2:
            raise ValueError("мало вариантов")

        ci = data.get("correct_index", None)
        correct_index = None
        if ci is not None:
            try:
                correct_index = int(ci)
            except Exception:
                correct_index = None
        if correct_index is not None and not (0 <= correct_index < len(options)):
            correct_index = None

        if not q:
            raise ValueError("пустой вопрос")
        return q, options, correct_index
    except Exception as e:
        raise RuntimeError(f"Не смог распарсить JSON от GPT: {e}. Ответ: {text[:400]}")


async def send_random_poll(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Авто-опрос (рандомный)."""
    q, opts = generate_random_poll()
    q = maybe_append_emoji(q)
    opts = [maybe_append_emoji_to_option(o) for o in opts]
    await context.bot.send_poll(
        chat_id=chat_id,
        question=q,
        options=opts,
        is_anonymous=True,
        allows_multiple_answers=False,
    )


# --------- МЕМЫ --------- #

def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Пытаемся найти шрифт с поддержкой кириллицы, иначе дефолт."""
    if os.path.exists(MEME_FONT_FILE):
        try:
            return ImageFont.truetype(MEME_FONT_FILE, size=size)
        except Exception as e:
            logger.error(f"Не удалось загрузить шрифт {MEME_FONT_FILE}: {e}")

    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue

    logger.warning("Не найден TTF-шрифт, используем дефолтный (кириллица может не отображаться).")
    return ImageFont.load_default()


def measure_text(draw: ImageDraw.ImageDraw, text: str, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for w in words:
        test = (current + " " + w).strip()
        width, _ = measure_text(draw, test, font)
        if width <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines or [""]


def draw_centered_text(draw: ImageDraw.ImageDraw, img_width: int, y: int, lines: List[str], font):
    for line in lines:
        w, h = measure_text(draw, line, font)
        x = (img_width - w) / 2
        outline_range = 2
        for dx in range(-outline_range, outline_range + 1):
            for dy in range(-outline_range, outline_range + 1):
                draw.text((x + dx, y + dy), line, font=font, fill="black")
        draw.text((x, y), line, font=font, fill="white")
        y += h + 5


def create_meme_image(top_text: str, bottom_text: str | None = None) -> BytesIO:
    candidates = sorted(glob.glob("mem*.jpg"))
    if not candidates:
        raise FileNotFoundError("Не найдено ни одного файла mem*.jpg рядом со script.py")

    path = random.choice(candidates)
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)

    top_text = (top_text or "").upper()
    bottom_text = (bottom_text or "").upper()

    base_font_size = max(24, img.height // 15)
    font = load_font(base_font_size)
    max_width = img.width - 40

    top_lines = wrap_text(draw, top_text, font, max_width) if top_text else []
    bottom_lines = wrap_text(draw, bottom_text, font, max_width) if bottom_text else []

    y_top = 10
    draw_centered_text(draw, img.width, y_top, top_lines, font)

    if bottom_lines:
        total_height = 0
        for line in bottom_lines:
            _, h = measure_text(draw, line, font)
            total_height += h + 5
        total_height -= 5
        y_bottom = img.height - total_height - 10
        draw_centered_text(draw, img.width, y_bottom, bottom_lines, font)

    bio = BytesIO()
    bio.name = "meme.jpg"
    img.save(bio, "JPEG")
    bio.seek(0)
    return bio


# --------- ДОП. ФУНКЦИИ --------- #

async def random_admin_insult(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    if random.random() >= RANDOM_ADMIN_INSULT_PROBABILITY:
        return

    admin = await get_random_admin(chat_id, context)
    if not admin:
        return

    mention = mention_html(admin.id, admin.full_name)
    text = maybe_append_emoji(f"{mention} шлюшка")

    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


async def morning_school_ping(context: ContextTypes.DEFAULT_TYPE):
    if CHANNEL_ID is None:
        return
    txt = maybe_append_emoji("все в школе мои сладкие?")
    await context.bot.send_message(chat_id=CHANNEL_ID, text=txt)


async def night_sleep_ping(context: ContextTypes.DEFAULT_TYPE):
    if CHANNEL_ID is None:
        return
    txt = maybe_append_emoji("все легли пупсы?")
    await context.bot.send_message(chat_id=CHANNEL_ID, text=txt)


# --------- ХЕНДЛЕРЫ --------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid = user.id if user else "unknown"

    await update.message.reply_text(
        "Привет! Я каналный мини-сглыпа 🤪\n\n"
        "• В канале читаю посты и иногда сам пишу бред.\n"
        "• /babble — сгенерить бред и отправить в канал.\n"
        "• /meme — сделать мем (mem*.jpg).\n"
        "• /say — написать от лица бота в канал.\n"
        "• /poll — опрос в канал (есть anon/multi/quiz/gpt).\n"
        f"• В канале: ответь на пост фразой «{MEME_TRIGGER}» — сделаю мем.\n\n"
        f"Твой user_id: {uid}\n"
        f"OWNER_ID в коде: {OWNER_ID}"
    )


async def channel_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    text = msg.text or msg.caption or ""

    # --- 1) Триггеры создания мема ---
    if text and msg.reply_to_message is not None:
        lowered = text.lower()
        if any(trigger in lowered for trigger in MEME_TRIGGERS):
            src = msg.reply_to_message
            src_text = src.text or src.caption or ""
            if not src_text:
                return
            try:
                bio = create_meme_image(src_text)
            except Exception as e:
                logger.error(f"Ошибка создания мема: {e}")
                return
            return await context.bot.send_photo(
                chat_id=msg.chat_id,
                photo=bio,
                reply_to_message_id=src.message_id,
            )

    # --- 2) Если не канал — выходим ---
    if msg.chat.type != "channel":
        return

    # --- 3) Добавляем текст в корпус + эмодзи ---
    add_tokens_from_message(msg)
    add_emojis_from_message(msg)

    # --- 4) Авто-опрос (10%) ---
    if random.random() < AUTO_POLL_PROBABILITY:
        try:
            await send_random_poll(msg.chat_id, context)
        except Exception as e:
            logger.error(f"Не удалось отправить poll: {e}")

    # --- 5) Авто-бред ---
    if random.random() < AUTO_POST_PROBABILITY:
        reply_text = maybe_append_emoji(make_babble_markov2())

        if random.random() < RANDOM_ADMIN_MENTION_PROBABILITY:
            admin = await get_random_admin(msg.chat_id, context)
            if admin is not None:
                mention = mention_html(admin.id, admin.full_name)
                reply_text = f"{mention} {reply_text}"

        if "<a href=" in reply_text:
            return await context.bot.send_message(
                chat_id=msg.chat_id,
                text=reply_text,
                parse_mode="HTML",
            )
        return await context.bot.send_message(chat_id=msg.chat_id, text=reply_text)

    # --- 6) Иногда только эмодзи ---
    if EMOJI_POOL and random.random() < AUTO_EMOJI_ONLY_PROBABILITY:
        e = pick_random_emoji()
        if e:
            if random.random() < 0.3:
                e2 = pick_random_emoji()
                e3 = pick_random_emoji()
                await context.bot.send_message(chat_id=msg.chat_id, text=f"{e}{e2}{e3}")
            else:
                await context.bot.send_message(chat_id=msg.chat_id, text=e)

    # --- 7) Случайно тегнуть админа ---
    await random_admin_insult(msg.chat_id, context)


async def babble_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("Эта команда только для админов.")
        return

    text = maybe_append_emoji(make_babble_markov2())
    target_chat_id = CHANNEL_ID or update.effective_chat.id

    if random.random() < RANDOM_ADMIN_MENTION_PROBABILITY:
        admin = await get_random_admin(target_chat_id, context)
        if admin is not None:
            mention = mention_html(admin.id, admin.full_name)
            text = f"{mention} {text}"
            await context.bot.send_message(
                chat_id=target_chat_id,
                text=text,
                parse_mode="HTML",
            )
            if target_chat_id != update.effective_chat.id:
                await update.message.reply_text("Отправил бред в канал.")
            return

    await context.bot.send_message(chat_id=target_chat_id, text=text)
    if target_chat_id != update.effective_chat.id:
        await update.message.reply_text("Отправил бред в канал.")


async def poll_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /poll [anon] [multi] [quiz] [gpt] <payload>

    Примеры:
    /poll вопрос | да | нет
    /poll anon вопрос | да | нет
    /poll multi вопрос | вариант1 | вариант2 | вариант3
    /poll quiz вопрос | *правильный | неправильный
    /poll gpt смешная тема про школу
    /poll gpt quiz тема про котов
    """
    if not await is_admin(update, context):
        await update.message.reply_text("Эта команда только для админов.")
        return

    target_chat_id = CHANNEL_ID or update.effective_chat.id
    args_text = " ".join(context.args) if context.args else ""

    flags, rest = parse_poll_flags_and_rest(args_text)

    is_anonymous = not flags["anon"]
    allows_multiple = flags["multi"]
    is_quiz = flags["quiz"]
    use_gpt = flags["gpt"]

    try:
        question: str
        options: List[str]
        correct_index: int | None = None

        parsed = parse_poll_payload(rest)

        if parsed:
            question, options, correct_index = parsed
            if is_quiz and correct_index is None:
                correct_index = random.randint(0, len(options) - 1)
        else:
            if use_gpt:
                try:
                    question, options, correct_index = generate_gpt_poll(rest, quiz=is_quiz)
                    if is_quiz and correct_index is None:
                        correct_index = random.randint(0, len(options) - 1)
                except Exception as e:
                    logger.error(f"GPT poll не сработал, делаю рандом: {e}")
                    question, options = generate_random_poll()
                    if is_quiz:
                        correct_index = random.randint(0, len(options) - 1)
                    try:
                        await update.message.reply_text(
                            "gpt не сработал (нет OPENAI_API_KEY или ошибка), сделал рандомный опрос."
                        )
                    except Exception:
                        pass
            else:
                question, options = generate_random_poll()
                if is_quiz:
                    correct_index = random.randint(0, len(options) - 1)

        question = maybe_append_emoji(question)
        options = [maybe_append_emoji_to_option(o) for o in options]

        if is_quiz and allows_multiple:
            allows_multiple = False

        poll_kwargs = dict(
            chat_id=target_chat_id,
            question=question[:290],
            options=[o[:95] for o in options][:10],
            is_anonymous=is_anonymous,
            allows_multiple_answers=allows_multiple,
        )

        if is_quiz:
            poll_kwargs["type"] = "quiz"
            poll_kwargs["correct_option_id"] = int(correct_index or 0)

        await context.bot.send_poll(**poll_kwargs)

    except Exception as e:
        logger.error(f"Не удалось отправить poll: {e}")
        await update.message.reply_text(f"Ошибка при отправке опроса: {e}")
        return

    if target_chat_id != update.effective_chat.id:
        await update.message.reply_text("Опрос отправлен в канал.")


async def osk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("Эта команда только для админов.")
        return

    target_chat_id = CHANNEL_ID or update.effective_chat.id
    admin = await get_random_admin(target_chat_id, context)
    if not admin:
        await update.message.reply_text("Не удалось найти админа.")
        return

    mention = mention_html(admin.id, admin.full_name)
    text = maybe_append_emoji(f"{mention} ты шлюшка")

    await context.bot.send_message(
        chat_id=target_chat_id,
        text=text,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    if target_chat_id != update.effective_chat.id:
        await update.message.reply_text("Отправлено.")


async def tagsay_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("Эта команда только для админов.")
        return

    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Использование:\n"
            "/tagsay <user_id> <текст в HTML>\n\n"
            "Пример:\n"
            "/tagsay 123456789 <b>Привет</b>, как дела?"
        )
        return

    target_chat_id = CHANNEL_ID or update.effective_chat.id
    user_id_str = context.args[0]
    try:
        user_id = int(user_id_str)
    except ValueError:
        await update.message.reply_text("user_id должен быть числом.\nПример: /tagsay 123456789 текст")
        return

    message_text = " ".join(context.args[1:])
    if not message_text:
        await update.message.reply_text("Нужно указать текст после user_id.")
        return

    try:
        member = await context.bot.get_chat_member(target_chat_id, user_id)
        display_name = member.user.full_name
    except Exception:
        display_name = user_id_str

    mention = mention_html(user_id, display_name)
    send_text = maybe_append_emoji(f"{mention} {message_text}")

    await context.bot.send_message(
        chat_id=target_chat_id,
        text=send_text,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    if target_chat_id != update.effective_chat.id:
        await update.message.reply_text("Отправлено в канал.")


async def say_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("Эта команда только для админов.")
        return

    text = " ".join(context.args) if context.args else ""
    if not text and update.message.reply_to_message:
        rep = update.message.reply_to_message
        text = rep.text or rep.caption or ""

    if not text:
        await update.message.reply_text(
            "Использование: /say <текст в HTML>\n\n"
            "Примеры:\n"
            "/say <b>Жирный текст</b>\n"
            "/say <a href=\"https://example.com\">Кликабельная ссылка</a>\n"
            "/say Привет, <i>курсив</i>!"
        )
        return

    target_chat_id = CHANNEL_ID or update.effective_chat.id

    if random.random() < RANDOM_ADMIN_MENTION_PROBABILITY:
        admin = await get_random_admin(target_chat_id, context)
        if admin is not None:
            mention = mention_html(admin.id, admin.full_name)
            text = f"{mention} {text}"

    text = maybe_append_emoji(text)

    await context.bot.send_message(
        chat_id=target_chat_id,
        text=text,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    if target_chat_id != update.effective_chat.id:
        await update.message.reply_text("Отправлено в канал.")


async def meme_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("Мемы может делать только админ 😎")
        return

    msg = update.message
    args_text = " ".join(context.args) if context.args else ""
    top_text = ""
    bottom_text = ""

    if msg.reply_to_message and not args_text:
        src = msg.reply_to_message
        top_text = src.text or src.caption or ""
        bottom_text = ""
    else:
        if "|" in args_text:
            parts = args_text.split("|", 1)
            top_text = parts[0].strip()
            bottom_text = parts[1].strip()
        else:
            top_text = args_text

    if not top_text and not bottom_text:
        await update.message.reply_text(
            "Использование:\n"
            "• Ответь на сообщение командой /meme — текст из сообщения станет подписью.\n"
            "• /meme Текст_сверху\n"
            "• /meme Текст_сверху | Текст_снизу"
        )
        return

    try:
        bio = create_meme_image(top_text, bottom_text)
    except FileNotFoundError as e:
        await update.message.reply_text(
            f"Ошибка: {e}\nУбедись, что mem*.jpg лежат рядом со script.py"
        )
        return

    await context.bot.send_photo(chat_id=msg.chat_id, photo=bio)


# --------- MAIN --------- #

def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "Не найден BOT_TOKEN в окружении.\n"
            "Задай: export BOT_TOKEN=\"...\""
        )

    if CHANNEL_ID is None:
        logger.warning("CHANNEL_ID не задан — /babble, /say, /poll не смогут писать в канал напрямую.")

    load_corpus_from_file()
    load_emojis()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    job_queue = app.job_queue
    if job_queue is None:
        logger.warning("JobQueue не доступен, планировщик не будет работать.")
    else:
        job_queue.run_daily(
            morning_school_ping,
            time=time(hour=9, minute=0, tzinfo=MOSCOW_TZ),
            name="morning_school_ping",
        )
        job_queue.run_daily(
            night_sleep_ping,
            time=time(hour=23, minute=0, tzinfo=MOSCOW_TZ),
            name="night_sleep_ping",
        )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("babble", babble_cmd))
    app.add_handler(CommandHandler("say", say_cmd))
    app.add_handler(CommandHandler("meme", meme_cmd))
    app.add_handler(CommandHandler("osk", osk_cmd))
    app.add_handler(CommandHandler("tagsay", tagsay_cmd))
    app.add_handler(CommandHandler("poll", poll_cmd))

    app.add_handler(MessageHandler(filters.ALL, channel_listener))

    logger.info("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
