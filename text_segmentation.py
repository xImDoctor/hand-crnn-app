# text_segmentation.py
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

def is_small_punctuation(box, avg_char_width, avg_char_height):
    x, y, w, h = box
    area = w * h
    is_very_small = w <= 6 or h <= 6 or area <= 50
    is_thin = w < avg_char_width * 0.25
    is_short = h < avg_char_height * 0.4
    is_tiny_area = area < avg_char_width * avg_char_height * 0.15
    return is_very_small or (is_thin and is_short) or is_tiny_area

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

def find_words_in_line(binary, line_box, distance_threshold=15):
    x, y, w, h = line_box
    line_img = binary[y:y+h, x:x+w]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    word_mask = cv2.dilate(line_img, kernel, iterations=1)

    contours, _ = cv2.findContours(word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = [cv2.boundingRect(c) for c in contours]
    if not raw_boxes:
        return []

    avg_char_width, avg_char_height = calculate_char_stats(raw_boxes)
    main_elements = []
    small_punctuation = []
    for bx, by, bw, bh in raw_boxes:
        if is_small_punctuation((bx, by, bw, bh), avg_char_width, avg_char_height):
            small_punctuation.append((bx, by, bw, bh))
        else:
            main_elements.append([bx, by, bw, bh])

    for px, py, pw, ph in small_punctuation:
        p_center_x = px + pw // 2
        p_center_y = py + ph // 2
        best_distance = float('inf')
        best_idx = -1
        for i, (mx, my, mw, mh) in enumerate(main_elements):
            m_center_y = my + mh // 2
            if abs(p_center_y - m_center_y) > avg_char_height * 0.4:
                continue
            if p_center_x < mx:
                distance = mx - (px + pw)
            elif p_center_x > mx + mw:
                distance = px - (mx + mw)
            else:
                distance = 0
            if distance <= avg_char_width * 0.3 and distance < best_distance:
                best_distance = distance
                best_idx = i
        if best_idx >= 0:
            mx, my, mw, mh = main_elements[best_idx]
            new_left = min(mx, px)
            new_top = min(my, py)
            new_right = max(mx + mw, px + pw)
            new_bottom = max(my + mh, py + ph)
            main_elements[best_idx] = [new_left, new_top, new_right - new_left, new_bottom - new_top]

    final_words = [(x + bx, y + by, bw, bh) for bx, by, bw, bh in main_elements]
    final_words = sorted(final_words, key=lambda b: b[0])
    return final_words

def segment_text(image_path):
    original, binary = preprocess_image(image_path)
    line_boxes = find_text_lines_by_clustering(binary)
    all_word_boxes = [find_words_in_line(binary, box) for box in line_boxes]
    return original, all_word_boxes
