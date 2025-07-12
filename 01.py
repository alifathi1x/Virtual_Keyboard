import time
from collections import deque

import cv2
import mediapipe as mp

# تنظیمات اولیه
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
)


def draw_rounded_rect(image, rect, color, corner_radius=15):
    """رسم مستطیل با گوشه‌های گرد"""
    x1, y1, x2, y2 = rect
    h, w = image.shape[:2]

    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

    if corner_radius > 0:
        cv2.ellipse(image, (x1 + corner_radius, y1 + corner_radius),
                    (corner_radius, corner_radius), 180, 0, 90, color, -1)
        cv2.ellipse(image, (x2 - corner_radius, y1 + corner_radius),
                    (corner_radius, corner_radius), 270, 0, 90, color, -1)
        cv2.ellipse(image, (x1 + corner_radius, y2 - corner_radius),
                    (corner_radius, corner_radius), 90, 0, 90, color, -1)
        cv2.ellipse(image, (x2 - corner_radius, y2 - corner_radius),
                    (corner_radius, corner_radius), 0, 0, 90, color, -1)

    cv2.rectangle(image, (x1 + corner_radius, y1),
                  (x2 - corner_radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + corner_radius),
                  (x2, y2 - corner_radius), color, -1)


class VirtualKeyboard:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.keys = []
        self.input_text = ""
        self.last_key_pressed = None
        self.key_press_times = {}
        self.finger_history = deque(maxlen=10)
        self.adaptive_threshold = 1.7
        self.last_entered_text = ""
        self.text_display_time = 0
        self.create_keyboard_layout()

    def create_keyboard_layout(self):
        keys_row1 = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
        keys_row2 = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';']
        keys_row3 = ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']
        keys_row4 = ['(', ')', '"']

        key_width = 60
        key_height = 40
        key_margin = 12
        corner_radius = 12
        start_y = self.frame_height - 400

        keyboard_width = 10 * (key_width + key_margin) - key_margin
        start_x = (self.frame_width - keyboard_width) // 2

        self.add_key_row(keys_row1, start_x, start_y, key_width, key_height, key_margin, corner_radius)

        row2_width = len(keys_row2) * (key_width + key_margin) - key_margin
        row2_start_x = start_x + (keyboard_width - row2_width) // 2
        self.add_key_row(keys_row2, row2_start_x, start_y + key_height + key_margin,
                         key_width, key_height, key_margin, corner_radius)

        row3_width = len(keys_row3) * (key_width + key_margin) - key_margin
        row3_start_x = start_x + (keyboard_width - row3_width) // 2
        self.add_key_row(keys_row3, row3_start_x, start_y + 2 * (key_height + key_margin),
                         key_width, key_height, key_margin, corner_radius)

        row4_width = len(keys_row4) * (key_width + key_margin) - key_margin
        row4_start_x = start_x + (keyboard_width - row4_width) // 2
        self.add_key_row(keys_row4, row4_start_x, start_y + 3 * (key_height + key_margin),
                         key_width, key_height, key_margin, corner_radius, special_color=(70, 50, 100))

        self.add_special_keys(start_x, keyboard_width, start_y, key_height, key_margin, corner_radius)

    def add_key_row(self, keys, start_x, start_y, width, height, margin, corner_radius, special_color=None):
        for i, key in enumerate(keys):
            x = start_x + i * (width + margin)
            color = special_color if special_color else (50, 50, 80)
            self.keys.append({
                'label': key,
                'rect': [x, start_y, x + width, start_y + height],
                'active': False,
                'progress': 0,
                'corner_radius': corner_radius,
                'color': color
            })

    def add_special_keys(self, start_x, keyboard_width, start_y, key_height, key_margin, corner_radius):
        special_width = 500
        special_start_x = start_x + (keyboard_width - special_width) // 2
        y_pos = start_y + 4 * (key_height + key_margin) + 10

        self.keys.append({
            'label': 'Space',
            'rect': [special_start_x, y_pos, special_start_x + 150, y_pos + key_height],
            'active': False,
            'progress': 0,
            'threshold_factor': 0.7,
            'corner_radius': corner_radius,
            'color': (80, 50, 50)
        })

        self.keys.append({
            'label': 'Del',
            'rect': [special_start_x + 160, y_pos, special_start_x + 240, y_pos + key_height],
            'active': False,
            'progress': 0,
            'corner_radius': corner_radius,
            'color': (80, 30, 30)
        })

        self.keys.append({
            'label': 'Enter',
            'rect': [special_start_x + 250, y_pos, special_start_x + 350, y_pos + key_height],
            'active': False,
            'progress': 0,
            'corner_radius': corner_radius,
            'color': (50, 80, 50)
        })

        self.keys.append({
            'label': 'Clear',
            'rect': [special_start_x + 360, y_pos, special_start_x + 440, y_pos + key_height],
            'active': False,
            'progress': 0,
            'corner_radius': corner_radius,
            'color': (80, 30, 30)
        })

    def update_adaptive_threshold(self, speed_factor):
        self.adaptive_threshold = max(0.8, min(2.5, 1.7 * speed_factor))

    def check_key_press(self, finger_pos, current_time):
        key_detected = None
        movement_factor = self.calculate_movement_factor(finger_pos)

        for key in self.keys:
            key['active'] = False
            key['progress'] = 0

            if self.is_in_key_with_tolerance(finger_pos, key, movement_factor):
                key['active'] = True
                key_detected = key['label']
                self.process_key_press(key, current_time)

        self.clean_inactive_keys(key_detected)

    def is_in_key_with_tolerance(self, finger_pos, key, tolerance_factor):
        x1, y1, x2, y2 = key['rect']
        fx, fy = finger_pos
        tolerance = 25 * tolerance_factor

        return (x1 - tolerance <= fx <= x2 + tolerance and
                y1 - tolerance <= fy <= y2 + tolerance)

    def process_key_press(self, key, current_time):
        threshold = self.adaptive_threshold
        if 'threshold_factor' in key:
            threshold *= key['threshold_factor']

        if key['label'] in self.key_press_times:
            hold_time = current_time - self.key_press_times[key['label']]
            key['progress'] = min(1.0, hold_time / threshold)

            if hold_time >= threshold:
                self.handle_key_press(key['label'], current_time)
                self.key_press_times[key['label']] = current_time
        else:
            self.key_press_times[key['label']] = current_time

    def clean_inactive_keys(self, active_key):
        inactive_keys = [k for k in self.key_press_times if k != active_key]
        for k in inactive_keys:
            self.key_press_times.pop(k, None)

    def calculate_movement_factor(self, finger_pos):
        self.finger_history.append(finger_pos)

        if len(self.finger_history) < 2:
            return 1.0

        total_distance = 0.0
        for i in range(1, len(self.finger_history)):
            x1, y1 = self.finger_history[i - 1]
            x2, y2 = self.finger_history[i]
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            total_distance += distance

        avg_speed = total_distance / (len(self.finger_history) - 1)
        return max(0.5, min(1.5, 1.0 / (avg_speed + 0.1)))

    def handle_key_press(self, key_label, current_time):
        if key_label == 'Del':
            self.input_text = self.input_text[:-1]
        elif key_label == 'Clear':
            self.input_text = ""
        elif key_label == 'Space':
            self.input_text += ' '
        elif key_label == 'Enter':
            self.last_entered_text = self.input_text
            self.text_display_time = current_time
            self.input_text = ""
        elif key_label == '(':
            self.input_text += '('
        elif key_label == ')':
            self.input_text += ')'
        elif key_label == '"':
            self.input_text += '"'
        else:
            self.input_text += key_label

        if self.last_key_pressed == key_label:
            self.adaptive_threshold = max(0.8, self.adaptive_threshold * 0.95)
        else:
            self.adaptive_threshold = min(2.5, self.adaptive_threshold * 1.05)

        self.last_key_pressed = key_label

    def draw_keyboard(self, frame):
        self.draw_keyboard_background(frame)

        for key in self.keys:
            self.draw_key(frame, key)

        self.draw_input_text(frame)
        self.draw_info_panel(frame)
        self.draw_entered_text(frame)

        return frame

    def draw_keyboard_background(self, frame):
        overlay = frame.copy()
        draw_rounded_rect(
            overlay,
            (50, self.frame_height - 430, self.frame_width - 50, self.frame_height - 80),
            (30, 30, 50),
            corner_radius=25
        )
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        shadow = frame.copy()
        cv2.rectangle(
            shadow,
            (50, self.frame_height - 430),
            (self.frame_width - 50, self.frame_height - 80),
            (0, 0, 0),
            -1
        )
        frame = cv2.addWeighted(shadow, 0.2, frame, 0.8, 0)

    def draw_key(self, frame, key):
        x1, y1, x2, y2 = key['rect']
        base_color = key.get('color', (50, 50, 80))
        corner_radius = key.get('corner_radius', 12)

        draw_rounded_rect(frame, (x1, y1, x2, y2), base_color, corner_radius)

        if key['active']:
            progress = key['progress']
            progress_height = int((y2 - y1) * progress)
            progress_color = (46 + int(progress * 50), 200 - int(progress * 50), 255)

            draw_rounded_rect(
                frame,
                (x1, y2 - progress_height, x2, y2),
                progress_color,
                corner_radius
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (86, 255, 245), 2)

        self.draw_key_label(frame, key)

    def draw_key_label(self, frame, key):
        x1, y1, x2, y2 = key['rect']
        text_color = (255, 255, 255) if key['active'] else (200, 200, 200)

        text_size = cv2.getTextSize(key['label'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2

        cv2.putText(frame, key['label'], (text_x + 2, text_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, key['label'], (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    def draw_input_text(self, frame):
        """ناحیه متن با پس زمینه سفید ساده"""
        cv2.rectangle(frame, (50, 30), (self.frame_width - 50, 100), (255, 255, 255), -1)
        cv2.rectangle(frame, (50, 30), (self.frame_width - 50, 100), (200, 200, 200), 2)

        text_y = 80
        for i, line in enumerate(self.input_text.split('\n')):
            cv2.putText(frame, line, (70, text_y + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def draw_entered_text(self, frame):
        """نمایش متن وارد شده با فونت سفید و موقعیت جدید (60 پیکسل پایین‌تر از کادر سفید)"""
        current_time = time.time()
        if current_time - self.text_display_time < 10 and self.last_entered_text:
            # نمایش متن 60 پیکسل پایین‌تر از کادر سفید ورودی متن (100 + 60 = 160)
            cv2.putText(frame, f"Entered: {self.last_entered_text}",
                        (70, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    def draw_info_panel(self, frame):
        threshold_text = f"Threshold: {self.adaptive_threshold:.2f}s"
        cv2.putText(frame, threshold_text, (self.frame_width - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.putText(frame, "Virtual Keyboard AR", (self.frame_width // 2 - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return

    keyboard = VirtualKeyboard(frame.shape[1], frame.shape[0])

    last_time = time.time()
    speed_history = deque(maxlen=30)
    prev_finger_pos = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        current_time = time.time()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(86, 255, 245), thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(50, 150, 200), thickness=2, circle_radius=2)
                )

                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                fx = int(index_finger.x * frame.shape[1])
                fy = int(index_finger.y * frame.shape[0])

                if prev_finger_pos:
                    distance = ((fx - prev_finger_pos[0]) ** 2 + (fy - prev_finger_pos[1]) ** 2) ** 0.5
                    speed = distance / (current_time - last_time)
                    speed_history.append(speed)

                prev_finger_pos = (fx, fy)
                last_time = current_time

                if speed_history:
                    avg_speed = sum(speed_history) / len(speed_history)
                    speed_factor = max(0.5, min(2.0, 1.0 / (avg_speed + 0.1)))
                    keyboard.update_adaptive_threshold(speed_factor)

                keyboard.check_key_press((fx, fy), current_time)

                cv2.circle(frame, (fx, fy), 15, (0, 255, 255), 2)
                cv2.circle(frame, (fx, fy), 8, (0, 200, 255), -1)

        frame = keyboard.draw_keyboard(frame)

        cv2.imshow('Virtual Keyboard AR', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
