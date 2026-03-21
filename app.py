import gradio as gr
from ultralytics import YOLO
import pandas as pd

model = YOLO('C:/Users/ЭЛЬДО/Downloads/best_8m.pt')


# DataFrame для сбора данных
df = pd.DataFrame({
    'Missing_hole': [],
    'Mouse_bite': [],
    'Open_circuit': [],
    'Short': [],
    'Spur': [],
    'Spurious_copper': [],
})

def table(results):
    arr = [0, 0, 0, 0, 0, 0]
    detect = results[0].boxes.cls.cpu().numpy() # Извлекаем классы обнаруженных объектов
    for i in range(len(detect)): # Проходим циклом по найденым дефектом и увеличиваем определённое место в списке
        arr[int(detect[i])] += 1
    return arr

def detect_objects_in_video(frame):
    if frame is None:
        return None
    # Запускаем инференцию YOLO
    results = model(frame, stream=False)
    # Получаем аннотированный кадр
    for r in results:
        annotated_frame = r.plot()

    return annotated_frame


# Функция для обработки изображения
def process_image(img):
    result = model.predict( # Предсказание дефектов
            source=img,
            conf=0.25,
            save=True,
            project="content/preict_itog",
            name="results",
            exist_ok=True
        )
    print(result)
    # Добавление новой строки
    df.loc[len(df)] = table(result)
    print(df)
    return result[0].plot()

with gr.Blocks() as calculator_page:
    gr.Markdown("# Ручной способ")
    with gr.Row():
        with gr.Column():
            # Поле для загрузки изображений
            input_img = gr.Image(
                label="Загрузите изображение",
                type="numpy"
            )
            # Кнопка для запуска обработки
            submit_btn = gr.Button("Обработать изображение", variant="primary")

        with gr.Column():
            output_img = gr.Image(label="Результат") # Поле для вывода результата

    # Связываем кнопку с функцией
    submit_btn.click(
        fn=process_image,
        inputs=input_img,
        outputs=output_img
    )

def new_table():
    return df

with gr.Blocks() as podrobno:
    gr.Markdown("# Отчёт")
    report_table = gr.DataFrame(label="Статистика дефектов", value=df)  # Инициализируем таблицу
    refresh_btn = gr.Button("Обновить отчёт") # Кнопка для обновления таблицы
    refresh_btn.click( # Связываем кнопку с функцией для обновления таблицы
        fn=new_table,
        inputs=[],
        outputs=report_table
    )

with gr.Blocks() as text_converter_page:
    with gr.Blocks(title="YOLOv8 Live Object Detection") as demo:
        gr.Markdown("# Детекция объектов в реальном времени с YOLOv8")
        gr.Markdown("Наведите камеру на объекты — модель YOLOv8 будет детектировать их в реальном времени.")

        with gr.Row():
            # Входной компонент — веб‑камера
            webcam_input = gr.Image(
                label="Видеопоток с веб‑камеры",
                type="numpy",
                streaming=True  # Включаем режим потоковой передачи
            )

            # Выходной компонент — результат детекции
            output_image = gr.Image(
                label="Результат детекции",
                type="numpy"
            )

        # Обработчик для live‑детекции
        webcam_input.stream(
            fn=detect_objects_in_video,
            inputs=webcam_input,
            outputs=output_image,
            time_limit=600,
            show_progress=False
        )

        gr.Markdown("**Инструкция:**\n1. Разрешите доступ к камере.\n2. Наведите камеру на объекты.\n3. Наблюдайте за детекцией в реальном времени.\n4. Нажмите 'Stop' для остановки.")



with gr.Blocks() as random_generator_page:
    gr.Markdown("# Учебный проект СВКК(Система визуального контроля качества)")
    gr.Markdown("## Анотация проетка")
    gr.Markdown("""Проект по разработке автоматизированной системы визуального контроля качества (СВКК) на базе нейросетевых технологий, которая позволит полностью заменить ручной визуальный контроль на производственной линии.

Название проекта: Разработка автоматизированной системы визуального контроля качества (СВКК) на базе нейросетевых технологий.

Цель проекта: Полная замена ручного визуального контроля на производственной линии посредством внедрения автоматизированной системы.

Ключевые преимущества готового решения:
-Высокоскоростной контроль: проверка изделий осуществляется в режиме реального времени, не замедляя производственный процесс.
-Повышенная точность: система демонстрирует более высокую эффективность в выявлении дефектов по сравнению с ручным контролем.
-Объективность оценки: результаты не зависят от субъективного мнения оператора или его усталости.
-Полная прослеживаемость: все выявленные дефекты автоматически фиксируются и привязываются к конкретному изделию, формируя подробный отчёт.
Результаты реализации:
В рамках проекта успешно разработана и внедрена СВКК, базирующаяся на нейросетевых алгоритмах. Система полностью соответствует заявленным целям и задачам.

Технологическое преимущество:
Разработанная модель не использует примитивное сравнение с эталонным образцом. Вместо этого она выполняет прямую детекцию дефектов (объектов) на изображении. Данный подход обеспечивает универсальность применения — система совместима с широким спектром типов печатных плат без необходимости глубокой перенастройки.""")
    gr.Markdown("## Инструкция по использованию")
    gr.Markdown("Для использования проекта необходимо загрузить изображение и нажать кнопку 'Обработать изображение'. Результат будет выведен в окне 'Результат'")

# Объединение страниц во вкладках
demo = gr.TabbedInterface(
    [calculator_page, text_converter_page, podrobno, random_generator_page],
    ["Ручной", "Через web-камеру", "Отчёт" ,"Подробности"]
).launch(debug=True)

if __name__ == "__main__":
    demo.launch(share=True)