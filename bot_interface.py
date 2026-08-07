import logging
from aiogram import Bot, F, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.utils.keyboard import InlineKeyboardBuilder

logger = logging.getLogger(__name__)

# Состояния для пошагового диалога
class SeismicFSM(StatesGroup):
    waiting_for_region_choice = State()
    waiting_for_point_decision = State()
    waiting_for_custom_coords = State()

def register_handlers(dp):
    """Регистрирует все обработчики диалога в диспетчере aiogram."""

    # 1. Нажатие кнопки "Да" после глобального отчета (выбор региона)
    @dp.callback_query(F.data == "regions_yes")
    async def regions_yes(callback: types.CallbackQuery, state: FSMContext):
        await state.set_state(SeismicFSM.waiting_for_region_choice)
        
        builder = InlineKeyboardBuilder()
        builder.button(text="🇮🇩 Индонезия", callback_data="reg_indonesia")
        builder.button(text="🇯🇵 Япония", callback_data="reg_japan")
        builder.button(text="❌ Выход", callback_data="reg_exit")
        builder.adjust(1)
        
        await callback.message.edit_text("📍 Выберите регион для углубленного анализа:", reply_markup=builder.as_markup())
        await callback.answer()

    # Нажатие кнопки "Нет" после глобального отчета
    @dp.callback_query(F.data == "regions_no")
    async def regions_no(callback: types.CallbackQuery, state: FSMContext):
        await callback.message.edit_text("✅ Глобальный прогноз завершен. Хорошего дня!")
        await state.clear()
        await callback.answer()

    # 2. Обработка выбора конкретного региона
    @dp.callback_query(SeismicFSM.waiting_for_region_choice, F.data.startswith("reg_"))
    async def process_region_choice(callback: types.CallbackQuery, state: FSMContext):
        data = callback.data
        if data == "reg_exit":
            await callback.message.edit_text("❌ Анализ отменен.")
            await state.clear()
            await callback.answer()
            return

        region_name = data.split("_")[1]
        await state.update_data(selected_region=region_name)
        await state.set_state(SeismicFSM.waiting_for_point_decision)

        builder = InlineKeyboardBuilder()
        builder.button(text="Да", callback_data="point_yes")
        builder.button(text="Нет", callback_data="point_no")
        builder.button(text="⬅️ Назад", callback_data="point_back")
        builder.button(text="❌ Выход", callback_data="point_exit")
        builder.adjust(2, 2)

        await callback.message.edit_text(
            f"Вы выбрали регион: **{region_name.capitalize()}**.\n\nНужна конкретная точка или город?",
            reply_markup=builder.as_markup(),
            parse_mode="Markdown"
        )
        await callback.answer()

    # 3. Обработка вопроса про точку или город
    @dp.callback_query(SeismicFSM.waiting_for_point_decision, F.data.startswith("point_"))
    async def process_point_decision(callback: types.CallbackQuery, state: FSMContext):
        action = callback.data.split("_")[1]

        if action == "exit":
            await callback.message.edit_text("❌ Сеанс анализа завершен.")
            await state.clear()
            await callback.answer()
            return

        if action == "back":
            # Возврат к выбору региона
            await state.set_state(SeismicFSM.waiting_for_region_choice)
            builder = InlineKeyboardBuilder()
            builder.button(text="🇮🇩 Индонезия", callback_data="reg_indonesia")
            builder.button(text="🇯🇵 Япония", callback_data="reg_japan")
            builder.button(text="❌ Выход", callback_data="reg_exit")
            builder.adjust(1)
            await callback.message.edit_text("📍 Выберите регион заново:", reply_markup=builder.as_markup())
            await callback.answer()
            return

        if action == "no":
            # Пользователь выбрал анализ ВСЕГО региона без точки
            user_data = await state.get_data()
            region = user_data.get("selected_region")
            await callback.message.edit_text(f"🔄 Запускаю региональный анализ и прогнозирование для: **{region}**...")
            # <--- ТУТ ВЫЗЫВАЙТЕ ВАШУ ФУНКЦИЮ РЕГИОНАЛЬНОГО АНАЛИЗА --->
            await state.clear()
            await callback.answer()
            return

        if action == "yes":
            # Переход к вводу координат/города
            await state.set_state(SeismicFSM.waiting_for_custom_coords)
            await callback.message.edit_text(
                "✍️ Пожалуйста, введите координаты (например: `-6.2, 106.8`) или название города текстовым сообщением:"
            )
            await callback.answer()

    # 4. Получение введенных координат или города текстом
    @dp.message(SeismicFSM.waiting_for_custom_coords, F.text)
    async def process_custom_coords(message: types.Message, state: FSMContext):
        user_input = message.text.strip()
        user_data = await state.get_data()
        region = user_data.get("selected_region")

        await message.answer(
            f"🎯 Получена точка/город: *{user_input}* (регион: {region}).\n🔄 Запускаю точечный анализ..."
        )
        # <--- ТУТ ВЫЗЫВАЙТЕ ВАШУ ФУНКЦИЮ ТОЧЕЧНОГО АНАЛИЗА --->
        await state.clear()
