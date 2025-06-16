# improved_text_segmentation.py
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return image, binary

def find_text_lines_by_clustering(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]
    centers_np = np.array(centers)

    if len(centers_np) < 2:
        return []

    clustering = DBSCAN(eps=25, min_samples=1).fit(centers_np[:, 1].reshape(-1, 1))
    labels = clustering.labels_

    lines = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        line_boxes = [boxes[i] for i in idx]

        x_min = min(b[0] for b in line_boxes)
        y_min = min(b[1] for b in line_boxes)
        x_max = max(b[0] + b[2] for b in line_boxes)
        y_max = max(b[1] + b[3] for b in line_boxes)

        lines.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return sorted(lines, key=lambda b: b[1])

def calculate_char_stats(boxes):
    if not boxes:
        return 10, 15
    widths = sorted([w for x, y, w, h in boxes])
    heights = sorted([h for x, y, w, h in boxes])
    start_idx = len(widths) // 10
    end_idx = len(widths) - start_idx
    avg_width = np.mean(widths[start_idx:end_idx]) if end_idx > start_idx else np.mean(widths)
    avg_height = np.mean(heights[start_idx:end_idx]) if end_idx > start_idx else np.mean(heights)
    return avg_width, avg_height

def merge_overlapping_boxes(boxes):
    """Объединяет перекрывающиеся или близко расположенные боксы"""
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda x: x[0])  # Сортируем по x
    merged = [list(boxes[0])]
    
    for current in boxes[1:]:
        last = merged[-1]
        
        # Проверяем пересечение или близость по горизонтали и вертикали
        horizontal_overlap = (current[0] <= last[0] + last[2] + 5)  # +5 пикселей буфер
        vertical_overlap = not (current[1] + current[3] < last[1] - 3 or 
                               last[1] + last[3] < current[1] - 3)
        
        if horizontal_overlap and vertical_overlap:
            # Объединяем боксы
            new_x = min(last[0], current[0])
            new_y = min(last[1], current[1])
            new_right = max(last[0] + last[2], current[0] + current[2])
            new_bottom = max(last[1] + last[3], current[1] + current[3])
            merged[-1] = [new_x, new_y, new_right - new_x, new_bottom - new_y]
        else:
            merged.append(list(current))
    
    return merged

def find_words_by_spacing_analysis(binary, line_box):
    """Улучшенный алгоритм поиска слов на основе анализа расстояний"""
    x, y, w, h = line_box
    line_img = binary[y:y+h, x:x+w]
    
    # Более агрессивная морфологическая обработка для соединения частей букв
    # Используем горизонтальное соединение
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    connected = cv2.morphologyEx(line_img, cv2.MORPH_CLOSE, horizontal_kernel)
    
    # Вертикальное соединение для букв с разрывами
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, vertical_kernel)
    
    # Найдем все контуры
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]
    
    if not boxes:
        return []
    
    # Объединяем перекрывающиеся боксы
    boxes = merge_overlapping_boxes(boxes)
    
    # Вычисляем статистики символов
    avg_char_width, avg_char_height = calculate_char_stats(boxes)
    
    # Анализ расстояний между боксами для группировки в слова
    boxes = sorted(boxes, key=lambda b: b[0])
    words = []
    current_word = [boxes[0]]
    
    for i in range(1, len(boxes)):
        prev_box = current_word[-1]
        curr_box = boxes[i]
        
        # Расстояние между концом предыдущего бокса и началом текущего
        gap = curr_box[0] - (prev_box[0] + prev_box[2])
        
        # Проверяем вертикальное выравнивание
        prev_center_y = prev_box[1] + prev_box[3] // 2
        curr_center_y = curr_box[1] + curr_box[3] // 2
        vertical_diff = abs(prev_center_y - curr_center_y)
        
        # Адаптивный порог для разделения слов
        # Учитываем размер символов и их высоту
        char_spacing_threshold = max(avg_char_width * 0.1, 2)  # Минимум 8 пикселей
        word_spacing_threshold = max(avg_char_width * 0.2, 4)  # Минимум 15 пикселей
        height_threshold = avg_char_height * 0.3
        
        # Условия для объединения в одно слово:
        # 1. Небольшой разрыв (меньше порога для разделения слов)
        # 2. Хорошее вертикальное выравнивание
        # 3. Или очень маленький разрыв (части одной буквы)
        should_merge = (
            (gap <= word_spacing_threshold and vertical_diff <= height_threshold) or
            gap <= char_spacing_threshold or
            gap < 0  # Перекрывающиеся боксы
        )
        
        if should_merge:
            current_word.append(curr_box)
        else:
            # Завершаем текущее слово и начинаем новое
            if current_word:
                words.append(current_word)
            current_word = [curr_box]
    
    # Добавляем последнее слово
    if current_word:
        words.append(current_word)
    
    # Создаем финальные боксы для слов
    final_words = []
    for word_boxes in words:
        if not word_boxes:
            continue
            
        # Объединяем все боксы слова в один
        min_x = min(box[0] for box in word_boxes)
        min_y = min(box[1] for box in word_boxes)
        max_x = max(box[0] + box[2] for box in word_boxes)
        max_y = max(box[1] + box[3] for box in word_boxes)
        
        # Добавляем небольшой паддинг
        pad = 6
        h_img, w_img = binary.shape
        
        x1 = max(min_x - pad, 0)
        y1 = max(min_y - pad, 0)
        x2 = min(max_x + pad, w_img)
        y2 = min(max_y + pad, h_img)
        
        # Преобразуем в глобальные координаты
        global_x = x + x1
        global_y = y + y1
        width = x2 - x1
        height = y2 - y1
        
        final_words.append((global_x, global_y, width, height))
    
    return sorted(final_words, key=lambda b: b[0])

def post_process_words(word_boxes, avg_char_width):
    """Дополнительная постобработка для объединения близких слов"""
    if len(word_boxes) <= 1:
        return word_boxes
    
    processed = []
    current_group = [word_boxes[0]]
    
    for i in range(1, len(word_boxes)):
        prev_word = current_group[-1]
        curr_word = word_boxes[i]
        
        # Расстояние между словами
        gap = curr_word[0] - (prev_word[0] + prev_word[2])
        
        # Если слова очень близко (возможно, разорванное слово)
        if gap <= avg_char_width * 0.8:
            current_group.append(curr_word)
        else:
            # Завершаем группу и начинаем новую
            if len(current_group) > 1:
                # Объединяем группу в одно слово
                min_x = min(w[0] for w in current_group)
                min_y = min(w[1] for w in current_group)
                max_x = max(w[0] + w[2] for w in current_group)
                max_y = max(w[1] + w[3] for w in current_group)
                processed.append((min_x, min_y, max_x - min_x, max_y - min_y))
            else:
                processed.extend(current_group)
            current_group = [curr_word]
    
    # Обрабатываем последнюю группу
    if len(current_group) > 1:
        min_x = min(w[0] for w in current_group)
        min_y = min(w[1] for w in current_group)
        max_x = max(w[0] + w[2] for w in current_group)
        max_y = max(w[1] + w[3] for w in current_group)
        processed.append((min_x, min_y, max_x - min_x, max_y - min_y))
    else:
        processed.extend(current_group)
    
    return processed

def segment_text(image_path):
    """Главная функция сегментации с улучшенным алгоритмом"""
    original, binary = preprocess_image(image_path)
    line_boxes = find_text_lines_by_clustering(binary)
    
    # Вычисляем средние размеры символов для всего изображения
    all_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_boxes = [cv2.boundingRect(c) for c in all_contours if cv2.contourArea(c) > 10]
    avg_char_width, avg_char_height = calculate_char_stats(all_boxes)
    
    all_word_boxes = []
    for line_box in line_boxes:
        words = find_words_by_spacing_analysis(binary, line_box)
        # Применяем постобработку для каждой линии
        words = post_process_words(words, avg_char_width)
        all_word_boxes.append(words)
    
    return original, all_word_boxes

# Дополнительная функция для визуализации результатов
def visualize_segmentation(image_path, save_path=None):
    """Визуализация результатов сегментации"""
    original, word_boxes = segment_text(image_path)
    
    # Создаем цветное изображение для визуализации
    if len(original.shape) == 2:
        vis_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = original.copy()
    
    # Рисуем боксы разными цветами для каждой линии
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    
    for line_idx, line_words in enumerate(word_boxes):
        color = colors[line_idx % len(colors)]
        for x, y, w, h in line_words:
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            # Добавляем номер слова
            cv2.putText(vis_img, f'L{line_idx+1}', (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img
