import os.path
import cv2
import glob
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

model = YOLO("weights/best.pt")

img_path = "data/sample_sequence"
img_files = glob.glob(os.path.join(img_path, "*.jpg"))

track_his = defaultdict(lambda : [])

previous_y_positions = {}
crossed_count = 0
counted_ids = set()

if not img_files:
    print("Not found")
else:
    frist_frame = cv2.imread(img_files[0])
    height, width, _ = frist_frame.shape

    LINE_Y = height - 100

    output_path = "output_sequence.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 24.0, (width, height))

    for img_file in img_files:
        frame = cv2.imread(img_file)

        result = model.track(frame, persist=True, verbose=False )
        annotated_frame = frame.copy()

        if result[0].boxes.id is not None:
            boxes_xyxy = result[0].boxes.xyxy.cpu()
            boxes_xywh = result[0].boxes.xywh.cpu()
            scores = result[0].boxes.conf.cpu()
            classes = result[0].boxes.cls.cpu()

            for box_xy, score, cls in zip(boxes_xyxy, scores, classes):
                if score < 0.6:
                    continue
                x1, y1, x2, y2 = map(int, box_xy)
                label = f"{model.names[int(cls)]} {score:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            track_ids = result[0].boxes.id.int().cpu().tolist()

            LINE_Y = int(height - 100)
            # Tracking and Counting
            for box_wh, track_id in zip(boxes_xywh, track_ids):
                x_center, y_center, w, h = box_wh
                current_y = float(y_center)

                if track_id in previous_y_positions:
                    prev_y = previous_y_positions[track_id]

                    if prev_y < LINE_Y and current_y >= LINE_Y:
                        if track_id not in counted_ids:
                            crossed_count += 1
                            counted_ids.add(track_id)
                    elif prev_y > LINE_Y and current_y <= LINE_Y:
                        if track_id not in counted_ids:
                            crossed_count += 1
                            counted_ids.add(track_id)

                previous_y_positions[track_id] = current_y

                x_center, y_center, w, h = box_wh
                track = track_his[track_id]
                track.append((float(x_center), float(y_center)))

                if len(track) > 60:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                cv2.polylines(annotated_frame, [points], False, (0, 255, 255), 2)
        cv2.putText(annotated_frame, str(crossed_count), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        out.write(annotated_frame)

    out.release()
