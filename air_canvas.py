
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

blue_lines = [deque(maxlen=1024)]
pink_lines = [deque(maxlen=1024)]
green_lines = [deque(maxlen=1024)]
red_lines = [deque(maxlen=1024)]

blue_ptr = 0
pink_ptr = 0
green_ptr = 0
red_ptr = 0

brush_colors = [(255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255)]
current_color = 0

canvas = np.ones((471, 636, 3), dtype=np.uint8) * 255
cv2.rectangle(canvas, (40, 1), (140, 65), (0, 0, 0), 2)
cv2.rectangle(canvas, (160, 1), (255, 65), (255, 0, 0), 2)
cv2.rectangle(canvas, (275, 1), (370, 65), (255, 0, 255), 2)
cv2.rectangle(canvas, (390, 1), (485, 65), (0, 255, 0), 2)
cv2.rectangle(canvas, (505, 1), (600, 65), (0, 0, 255), 2)

cv2.putText(canvas, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(canvas, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(canvas, "PINK", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(canvas, "GREEN", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(canvas, "RED", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

cv2.namedWindow('Virtual Canvas', cv2.WINDOW_AUTOSIZE)

mp_hands = mp.solutions.hands
hand_tracker = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, _ = frame.shape

    cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    cv2.rectangle(frame, (275, 1), (370, 65), (255, 0, 255), 2)
    cv2.rectangle(frame, (390, 1), (485, 65), (0, 255, 0), 2)
    cv2.rectangle(frame, (505, 1), (600, 65), (0, 0, 255), 2)

    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "PINK", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "GREEN", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "RED", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    results = hand_tracker.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                px, py = int(lm.x * width), int(lm.y * height)
                lm_list.append((px, py))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = lm_list[8]
            thumb_tip = lm_list[4]

            cv2.circle(frame, index_tip, 5, (0, 255, 0), -1)

            if (thumb_tip[1] - index_tip[1]) < 30:
                blue_lines.append(deque(maxlen=512))
                pink_lines.append(deque(maxlen=512))
                green_lines.append(deque(maxlen=512))
                red_lines.append(deque(maxlen=512))
                blue_ptr += 1
                pink_ptr += 1
                green_ptr += 1
                red_ptr += 1
            elif index_tip[1] <= 65:
                if 40 <= index_tip[0] <= 140:
                    blue_lines = [deque(maxlen=512)]
                    pink_lines = [deque(maxlen=512)]
                    green_lines = [deque(maxlen=512)]
                    red_lines = [deque(maxlen=512)]
                    blue_ptr = pink_ptr = green_ptr = red_ptr = 0
                    canvas[67:, :, :] = 255
                elif 160 <= index_tip[0] <= 255:
                    current_color = 0
                elif 275 <= index_tip[0] <= 370:
                    current_color = 1
                elif 390 <= index_tip[0] <= 485:
                    current_color = 2
                elif 505 <= index_tip[0] <= 600:
                    current_color = 3
            else:
                if current_color == 0:
                    blue_lines[blue_ptr].appendleft(index_tip)
                elif current_color == 1:
                    pink_lines[pink_ptr].appendleft(index_tip)
                elif current_color == 2:
                    green_lines[green_ptr].appendleft(index_tip)
                elif current_color == 3:
                    red_lines[red_ptr].appendleft(index_tip)
    else:
        blue_lines.append(deque(maxlen=512))
        pink_lines.append(deque(maxlen=512))
        green_lines.append(deque(maxlen=512))
        red_lines.append(deque(maxlen=512))
        blue_ptr += 1
        pink_ptr += 1
        green_ptr += 1
        red_ptr += 1

    all_lines = [blue_lines, pink_lines, green_lines, red_lines]

    for color_id, color_lines in enumerate(all_lines):
        for stroke in color_lines:
            for i in range(1, len(stroke)):
                if stroke[i - 1] is None or stroke[i] is None:
                    continue
                cv2.line(frame, stroke[i - 1], stroke[i], brush_colors[color_id], 2)
                cv2.line(canvas, stroke[i - 1], stroke[i], brush_colors[color_id], 2)

    cv2.imshow("Live Feed", frame)
    cv2.imshow("Virtual Canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
