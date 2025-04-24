# import numpy as np
# import tracker
# from detector import Detector
# import cv2
# import os
#
# # 创建保存目录
# save_dir = 'vehicle_image'
# os.makedirs(save_dir, exist_ok=True)
#
# cross_count = 0
# saved_ids = set()  # 记录已保存的 track_id
#
# if __name__ == '__main__':
#     detector = Detector()
#     capture = cv2.VideoCapture('./video/vehicle.mp4')
#
#     # 定义“上行”方向的多边形顶点
#     list_pts_up = [
#         [62, 447], [85, 573], [463, 664],
#         [959, 606], [1758, 743], [1760, 637],
#         [983, 521], [462, 579],
#         [157, 517], [125, 431]
#     ]
#     # 生成 mask 用于逻辑判断
#     mask_template = np.zeros((1280, 960), dtype=np.uint8)
#     pts_up = np.array([list_pts_up], dtype=np.int32)
#     poly_up_mask = cv2.fillPoly(mask_template.copy(), pts_up, color=1)
#     poly_up_mask = cv2.resize(poly_up_mask, (1280, 960), interpolation=cv2.INTER_NEAREST)
#
#     while True:
#         ret, im = capture.read()
#         if not ret:
#             break
#
#         # 缩放到与 mask 相同分辨率
#         im = cv2.resize(im, (1280, 960))
#         h, w = im.shape[:2]
#
#         # 检测并跟踪
#         bboxes = detector.detect(im)
#         tracks = tracker.update(bboxes, im) if bboxes else []
#
#         # 不绘制跟踪框，直接使用原图作为输出画面
#         output = im.copy()
#
#         # 可视化多边形轮廓
#         cv2.polylines(
#             output,
#             pts_up,
#             isClosed=True,
#             color=(0, 200, 200),
#             thickness=2,
#             lineType=cv2.LINE_AA
#         )
#
#         # 遍历每条轨迹，判断撞线并保存
#         for x1, y1, x2, y2, label, track_id in tracks:
#             # 底部中心点
#             cx = int((x1 + x2) / 2)
#             cy = int(y2)
#
#             # 判断是否进入多边形内部
#             if poly_up_mask[cy, cx] == 1 and track_id not in saved_ids:
#                 pad_h = 5
#                 pad_w = 20
#                 x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
#                 x1p = max(0, x1i - pad_w)
#                 y1p = max(0, y1i - pad_h)
#                 x2p = min(w, x2i + pad_w)
#                 y2p = min(h, y2i + pad_h)
#
#                 cross_count += 1
#                 crop = im[y1p:y2p, x1p:x2p]
#                 save_path = os.path.join(save_dir, f"{cross_count:03d}.jpg")
#                 cv2.imwrite(save_path, crop)
#                 print(f"Saved up-cross image: {save_path}")
#
#                 saved_ids.add(track_id)
#                 cv2.putText(
#                     output,
#                     f"SAVED:{cross_count:03d}",
#                     (x1i, y1i - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 0, 255),
#                     2
#                 )
#
#         # 显示保存总数
#         cv2.putText(
#             output,
#             f"Count: {cross_count}",
#             (20, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (255, 255, 255),
#             2
#         )
#
#         cv2.imshow('demo', output)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     capture.release()
#     cv2.destroyAllWindows()


# vehicle_detector.py
import os
import numpy as np
import tracker
from detector import Detector
import cv2
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/qt/plugins/platforms"


def run_vehicle_detection(video_path: str, save_dir: str = 'vehicle_image'):
    os.makedirs(save_dir, exist_ok=True)

    detector = Detector()
    capture = cv2.VideoCapture(video_path)

    cross_count = 0
    saved_ids = set()

    # 多边形区域设置
    list_pts_up = [
        [62, 447], [85, 573], [463, 664],
        [959, 606], [1758, 743], [1760, 637],
        [983, 521], [462, 579],
        [157, 517], [125, 431]
    ]
    mask_template = np.zeros((1280, 960), dtype=np.uint8)
    pts_up = np.array([list_pts_up], dtype=np.int32)
    poly_up_mask = cv2.fillPoly(mask_template.copy(), pts_up, color=1)
    poly_up_mask = cv2.resize(poly_up_mask, (1280, 960), interpolation=cv2.INTER_NEAREST)

    while True:
        ret, im = capture.read()
        if not ret:
            break

        im = cv2.resize(im, (1280, 960))
        h, w = im.shape[:2]

        bboxes = detector.detect(im)
        tracks = tracker.update(bboxes, im) if bboxes else []

        for x1, y1, x2, y2, label, track_id in tracks:
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            if poly_up_mask[cy, cx] == 1 and track_id not in saved_ids:
                pad_h = 5
                pad_w = 20
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                x1p = max(0, x1i - pad_w)
                y1p = max(0, y1i - pad_h)
                x2p = min(w, x2i + pad_w)
                y2p = min(h, y2i + pad_h)

                cross_count += 1
                crop = im[y1p:y2p, x1p:x2p]
                save_path = os.path.join(save_dir, f"{cross_count:03d}.jpg")
                cv2.imwrite(save_path, crop)
                print(f"Saved up-cross image: {save_path}")
                saved_ids.add(track_id)

    capture.release()
    cv2.destroyAllWindows()
    return cross_count
