from aiogram import Bot, Dispatcher, F
from aiogram.types import CallbackQuery
from aiogram.filters import Command
import os

bot = Bot(token=os.environ.get('TELEGRAM_BOT_TOKEN'))
dp = Dispatcher()

# Обработка нажатия "Да"
@dp.callback_query(F.data == "ask_regions")
async def show_regions(callback: CallbackQuery):
    # Тут выводим список регионов кнопками
    # ... логика отображения регионов ...
    await callback.message.edit_text("📍 Выберите регион:", reply_markup=...)

# Обработка выбора региона
@dp.callback_query(F.data.startswith("reg_"))
async def choose_point_or_city(callback: CallbackQuery):
    # Вопрос: нужна ли точка?
    # ... кнопки "Да", "Нет", "Назад" ...
    await callback.message.edit_text("Нужна конкретная точка или город?", reply_markup=...)

# Запуск polling в отдельном потоке или основном цикле
async def main():
    await dp.start_polling(bot)
  
