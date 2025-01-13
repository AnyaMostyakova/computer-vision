import cv2
import numpy as np

def process_frame(frame, mode):
    # Преобразование кадра в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if mode == '1':
        # Диапазон для желтого цвета (режим видеофайла)
        lower_color = np.array([20, 100, 100])  # Нижняя граница желтого
        upper_color = np.array([30, 255, 255])  # Верхняя граница желтого
    elif mode == '2':
        # Диапазон для черного цвета (режим камеры)
        lower_color = np.array([0, 0, 0])  # Нижняя граница черного
        upper_color = np.array([180, 255, 50])  # Верхняя граница черного

    # Создание маски для выбранного цвета
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Поиск контуров на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Нахождение наибольшего контура
        largest_contour = max(contours, key=cv2.contourArea)

        # Вычисление моментов для нахождения центра масс
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Отрисовка контура и центра масс
        if mode == '1':
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)  # Желтый контур
        elif mode == '2':
            cv2.drawContours(frame, [largest_contour], -1, (0, 0, 255), 2)  # Красный контур
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)  # Красный центр масс

        # Вывод координат центра масс в кадр
        cv2.putText(frame, f"Center: ({cX}, {cY})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Сообщение об отсутствии объекта
        cv2.putText(frame, "No object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def main():
    # Выбор режима работы
    mode = input("Выберите режим работы (1 - видеофайл, 2 - камера): ")

    if mode == '1':
        # Режим обработки видеофайла
        video_path = r"C:\Users\runga\Downloads\catball.mp4"  # Путь к видеофайлу
        cap = cv2.VideoCapture(video_path)
    elif mode == '2':
        # Режим обработки видеопотока с камеры
        cap = cv2.VideoCapture(0)
    else:
        print("Неверный режим. Завершение программы.")
        return

    if not cap.isOpened():
        print("Ошибка открытия видеофайла или камеры.")
        return

    # Установка скорости воспроизведения (в миллисекундах)
    delay = 30  # По умолчанию 30 мс (нормальная скорость)
    print("Используйте клавиши '+' и '-' для изменения скорости воспроизведения.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка кадра
        processed_frame = process_frame(frame, mode)

        # Отображение результата
        cv2.imshow('Frame', processed_frame)

        # Обработка клавиш для изменения скорости
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):  # Выход по нажатию 'q'
            break
        elif key == ord('+'):  # Увеличить скорость (уменьшить задержку)
            delay = max(1, delay - 5)  # Минимальная задержка 1 мс
            delay += 5  # Увеличение задержки

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()