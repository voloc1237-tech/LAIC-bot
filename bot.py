import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Ваши функции анализа (из main.py) – но они должны быть адаптированы
from laic_analysis import run_analysis  # предположим, что это ваш анализатор

TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')  # можно не использовать, если бот работает в группах/личке

logging.basicConfig(level=logging.INFO)

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот для мониторинга землетрясений.")

# Команда /analyze – запускает анализ и отправляет отчёт
async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Запускаем анализ (может быть долгим, но в синхронном виде заблокирует бота – лучше асинхронно)
    # Для упрощения можно сделать в отдельном потоке, но здесь для примера синхронно
    results = run_analysis()  # возвращает список результатов
    from telegram_sender import TelegramSender
    sender = TelegramSender()
    await sender.send_report(results)  # отправляет отчёт с кнопками

# Обработчик нажатия на кнопку
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "regions_yes":
        # Здесь можно отправить прогноз по регионам или запросить данные у DeepSeek
        await query.edit_message_text("Отлично! Сейчас я запрошу прогноз по регионам...")
        # Пример вызова DeepSeek (можно позже)
        # response = call_deepseek_api(...)
        # await query.edit_message_text(response)
    elif data == "regions_no":
        await query.edit_message_text("Хорошо, в следующий раз.")

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CallbackQueryHandler(button_handler))  # ловит нажатия кнопок
    app.run_polling()  # бесконечный цикл

if __name__ == "__main__":
    main()
