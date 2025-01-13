import cv2
import time

# Загрузка предобученных каскадов Хаара
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Инициализация видеозахвата
cap = cv2.VideoCapture(0)

# Переменные для расчета FPS
prev_time = 0
fps_list = []  # Список для хранения значений FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в grayscale для детекции
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Обработка каждого обнаруженного лица
    for (x, y, w, h) in faces:
        # Рамка вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Область лица для поиска глаз (верхняя половина)
        roi_gray_eyes = gray[y:y + h // 2, x:x + w]
        roi_color_eyes = frame[y:y + h // 2, x:x + w]

        # Детекция глаз в верхней половине лица
        eyes = eye_cascade.detectMultiScale(roi_gray_eyes, scaleFactor=1.1, minNeighbors=5)

        # Фильтрация глаз по размеру и положению
        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            # Проверка, что глаз находится в верхней половине лица
            if ey < h // 2:
                valid_eyes.append((ex, ey, ew, eh))

        # Используем только отфильтрованные глаза
        eyes = valid_eyes

        if len(eyes) < 2:  # Если хотя бы 1 глаз не обнаружен
            cv2.putText(frame, "Open your eyes", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Рамки вокруг глаз
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color_eyes, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Область лица для поиска улыбки (нижняя половина)
        roi_gray_smile = gray[y + h // 2:y + h, x:x + w]
        roi_color_smile = frame[y + h // 2:y + h, x:x + w]

        # Детекция улыбки
        smiles = smile_cascade.detectMultiScale(roi_gray_smile, scaleFactor=1.8, minNeighbors=20)
        if len(smiles) == 0:  # Если улыбка не обнаружена
            cv2.putText(frame, "Smile", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Рамки вокруг улыбки
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color_smile, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

    # Расчет FPS
    curr_time = time.time()
    time_diff = curr_time - prev_time
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_time = curr_time
    fps_list.append(fps)  # Сохраняем значение FPS для расчета среднего

    # Вывод FPS на кадр
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Отображение кадра
    cv2.imshow('Face Detection', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

# Расчет среднего FPS
average_fps = sum(fps_list) / len(fps_list) if fps_list else 0
print(f"Среднее значение FPS: {average_fps:.2f}")