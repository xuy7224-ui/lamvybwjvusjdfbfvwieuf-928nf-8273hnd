import json
import logging
import glob
import os
from telegram.helpers import mention_html
import random
import re
from io import BytesIO
from typing import List, Dict, Tuple

from PIL import Image, ImageDraw, ImageFont
from telegram import Update, Message
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ================== НАСТРОЙКИ ==================

# Токен бота от BotFather
BOT_TOKEN = "7901201601:AAFg96v9MY9nr4I3PRgBH4_IHnhu6YRF3u4"

# ID владельца, который может использовать функционал бота В ЛС
OWNER_ID = 7877092881

# ID канала, в который бот должен писать (/babble, /say, авто-бред)
# Пример: -1001234567890123
CHANNEL_ID = -1003009758716  # <<< ЗАМЕНИ на id своего канала

# Файл, где храним корпус токенов (слова + знаки)
CORPUS_FILE = "corpus_words.json"

# Вероятность, что бот сам ответит в канал бредом после нового поста
AUTO_POST_PROBABILITY = 0.18  # 0.15 = 15% случаев

# Вероятность, что бред будет адресован какому-то рандомному админу
RANDOM_ADMIN_MENTION_PROBABILITY = 0.3  # 0.3 = 30% случаев

# Триггер-фраза для мема в канале (ответом на сообщение)
MEME_TRIGGER = "сделай меме"

# Имя TTF-шрифта с кириллицей (должен лежать рядом со script.py)
MEME_FONT_FILE = "meme_font.ttf"

PUNCT = ".,!?"

# ===============================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Корпус токенов (слова/пунктуация)
CORPUS_TOKENS: List[str] = []

# Марковская цепь 2-го порядка: (w1, w2) -> [w3, w3, ...]
MARKOV2: Dict[Tuple[str, str], List[str]] = {}


# --------- ВСПОМОГАТЕЛЬНОЕ ---------

def tokenize(text: str) -> List[str]:
    """
    Разбиваем текст на токены:
    - слова/числа
    - отдельные знаки пунктуации . , ! ?
    """
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

    # В ЛС только владельцу разрешаем
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


def deny_if_not_owner_private(update: Update) -> bool:
    """
    Возвращает True, если нужно ОТКАЗАТЬ пользователю в ЛС.
    Используем в /start, /babble, /say, /meme.
    """
    chat = update.effective_chat
    user = update.effective_user
    if chat and chat.type == "private":
        if not user or user.id != OWNER_ID:
            # Можно ничего не отвечать, но давай скажем честно
            try:
                update.message.reply_text(
                    "Доступ к функционалу бота в ЛС только у владельца."
                )
            except Exception:
                pass
            return True
    return False


def update_markov_with_sequence(seq: List[str]):
    """Обновляем марковскую цепь новыми токенами подряд (2-й порядок)."""
    global MARKOV2

    if not seq:
        return

    # Связь с "хвостом" общего корпуса: берем последние два токена
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
    """Генерим текст по марковской цепи 2-го порядка."""

    # Если длина не задана — выбираем случайную от 1 до 13
    if max_tokens is None:
        max_tokens = random.randint(1, 13)

    if len(CORPUS_TOKENS) < 3 or not MARKOV2:
        return "Пока мало данных для марковской магии. Напишите что-нибудь в канал."

    start_pair = pick_start_pair()
    if not start_pair:
        return "Не удалось выбрать стартовую пару токенов."

    w1, w2 = start_pair
    tokens = [w1, w2]

    # Генерируем в пределах max_tokens
    while len(tokens) < max_tokens:
        key = (tokens[-2], tokens[-1])
        candidates = MARKOV2.get(key)
        if not candidates:
            break

        nxt = random.choice(candidates)

        # избегаем двойной пунктуации
        if nxt in PUNCT and tokens[-1] in PUNCT:
            continue

        tokens.append(nxt)

    # Обрезаем лишнее, если вдруг вышло больше
    tokens = tokens[:max_tokens]

    return tokens_to_text(tokens)

def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Пытаемся найти шрифт с поддержкой кириллицы, иначе дефолт."""
    # 1) Пробуем пользовательский шрифт в файле meme_font.ttf
    if os.path.exists(MEME_FONT_FILE):
        try:
            return ImageFont.truetype(MEME_FONT_FILE, size=size)
        except Exception as e:
            logger.error(f"Не удалось загрузить шрифт {MEME_FONT_FILE}: {e}")

    # 2) Пробуем системные варианты
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

    # 3) fallback — дефолтный (может плохо поддерживать кириллицу)
    logger.warning("Не найден TTF-шрифт, используем дефолтный (кириллица может не отображаться).")
    return ImageFont.load_default()


def measure_text(draw: ImageDraw.ImageDraw, text: str, font):
    """Возвращает (width, height) текста с учётом текущего шрифта."""
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    """Перенос строк по ширине картинки."""
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


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    img_width: int,
    y: int,
    lines: List[str],
    font,
):
    """Рисуем текст с обводкой по центру."""
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
    """Создаем мем на основе любого mem*.jpg, который реально есть в папке."""
    # Ищем все файлы формата mem*.jpg
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


# --------- ХЕНДЛЕРЫ ---------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if deny_if_not_owner_private(update):
        return

    await update.message.reply_text(
        "Привет! Я каналный мини-сглыпа 🤪\n\n"
        "• В канале читаю посты и иногда сам пишу бред.\n"
        "• /babble — сгенерить бред и отправить в канал.\n"
        "• /meme — сделать мем (mem1-5.jpg).\n"
        "• /say — написать от лица бота в канал.\n"
        f"• В канале: ответь на пост фразой «{MEME_TRIGGER}» — сделаю мем."
    )


async def channel_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Ловим все сообщения, но:
    - если это триггер "сделай меме"/"создай меме" как ответ -> делаем мем
    - если это канал -> добавляем в корпус + иногда пишем бред
    """
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

            await context.bot.send_photo(
                chat_id=msg.chat_id,
                photo=bio,
                reply_to_message_id=src.message_id,
            )
            return

    # --- 2) Если не канал — выходим ---
    if msg.chat.type != "channel":
        return

    # --- 3) Добавляем текст в корпус ---
    add_tokens_from_message(msg)

    # --- 4) Шанс отправить бред ---
    if random.random() < AUTO_POST_PROBABILITY:
        reply_text = make_babble_markov2()

        # шанс упоминания рандомного админа
        if random.random() < RANDOM_ADMIN_MENTION_PROBABILITY:
            admin = await get_random_admin(msg.chat_id, context)
            if admin is not None:
                mention = mention_html(admin.id, admin.full_name)
                reply_text = f"{mention} {reply_text}"

                await context.bot.send_message(
                    chat_id=msg.chat_id,
                    text=reply_text,
                    parse_mode="HTML",
                )
                return

        # обычный бред без упоминания
        await context.bot.send_message(chat_id=msg.chat_id, text=reply_text)


async def babble_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Генерация бреда и отправка в канал."""
    if not await is_admin(update, context):
        await update.message.reply_text("Эта команда только для админов.")
        return

    text = make_babble_markov2()
    target_chat_id = CHANNEL_ID or update.effective_chat.id

    # иногда упоминать рандомного админа
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
            return

    # обычный бред
    await context.bot.send_message(chat_id=target_chat_id, text=text)

    if target_chat_id != update.effective_chat.id:
        await update.message.reply_text("Отправил бред в канал.")




async def say_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пишет сообщение от лица бота в канал."""
    if not await is_admin(update, context):
        await update.message.reply_text("Эта команда только для админов.")
        return

    text = " ".join(context.args) if context.args else ""
    if not text and update.message.reply_to_message:
        rep = update.message.reply_to_message
        text = rep.text or rep.caption or ""

    if not text:
        await update.message.reply_text("Использование: /say текст")
        return

    target_chat_id = CHANNEL_ID or update.effective_chat.id

    # иногда упоминать рандомного админа
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
            return

    # обычная отправка
    await context.bot.send_message(chat_id=target_chat_id, text=text)

    if target_chat_id != update.effective_chat.id:
        await update.message.reply_text("Отправлено в канал.")


async def meme_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Создать мем (команда в личке/группе)."""
    if deny_if_not_owner_private(update):
        return

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
            f"Ошибка: {e}\nУбедись, что mem1.jpg..mem5.jpg лежат рядом со script.py"
        )
        return

    await context.bot.send_photo(chat_id=msg.chat_id, photo=bio)


# --------- MAIN ---------

def main():
    if not BOT_TOKEN or BOT_TOKEN in ("PASTE_YOUR_TOKEN_HERE", "PUT_YOUR_TOKEN_HERE"):
        raise RuntimeError("Поставь настоящий токен бота в BOT_TOKEN")

    if CHANNEL_ID is None:
        logger.warning("CHANNEL_ID не задан — /babble и /say не смогут писать в канал напрямую.")

    load_corpus_from_file()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Команды (в личке или группе)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("babble", babble_cmd))
    app.add_handler(CommandHandler("say", say_cmd))
    app.add_handler(CommandHandler("meme", meme_cmd))

    # Ловим сообщения ИЗ КАНАЛОВ
    channel_filter = filters.ChatType.CHANNEL
    app.add_handler(MessageHandler(channel_filter, channel_listener))

    logger.info("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()



