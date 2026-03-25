import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict


def process_video(model_path, video_path, output_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video tại đường dẫn '{video_path}'")
        return

    track_his = defaultdict(lambda:[])

    previous_y_positions = {}
    crossed_count = 0
    counted_ids = set()

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, 24.0, (int(width), int(height)))

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        result = model.track(frame, persist=True, verbose=False, conf=0.6)[0]
        annotated_frame = frame.copy()

        if result.boxes is not None and result.boxes.id is not None:
            boxes_xyxy = result.boxes.xyxy.cpu()
            boxes_xywh = result.boxes.xywh.cpu()
            scores = result.boxes.conf.cpu()
            classes = result.boxes.cls.cpu()

            #vẽ bbox
            for box_xy, score, cls in zip(boxes_xyxy, scores, classes):
                x1, y1, x2, y2 = map(int, box_xy)
                label = f"{model.names[int(cls)]} {score:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 3)

            track_ids = result.boxes.id.int().cpu().tolist()

            LINE_Y = int(height - 100)
            # Tracking and Counting
            for box_wh, track_id in zip(boxes_xywh, track_ids):
                x_center, y_center, w, h = box_wh
                current_y = float(y_center)

                if track_id in previous_y_positions:
                    prev_y = previous_y_positions[track_id]

                    if (
                        (prev_y < LINE_Y and current_y >= LINE_Y) or
                        (prev_y > LINE_Y and current_y <= LINE_Y)
                    ):
                        if track_id not in counted_ids:
                            crossed_count += 1
                            counted_ids.add(track_id)

                previous_y_positions[track_id] = current_y

                track = track_his[track_id]
                track.append((float(x_center), float(y_center)))

                if len(track) > 60:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                cv2.polylines(annotated_frame, [points], False, (0, 255, 255), 2)
            cv2.putText(annotated_frame, str(crossed_count), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 4)
            out.write(annotated_frame)

        cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    MODEL_WEIGHTS = "weights/best.pt"
    INPUT_VIDEO = "data/sample_video.mp4"
    OUTPUT_VIDEO = "output_video.mp4"

    process_video(MODEL_WEIGHTS, INPUT_VIDEO, OUTPUT_VIDEO)
