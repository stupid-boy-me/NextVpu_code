from tqdm import tqdm
from glob import glob
import os, cv2, shutil, psutil
import numpy as np
from copy import deepcopy

from utils import *

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from algorithm import *

psutil.Process(os.getpid()).cpu_affinity([0, 1, 2, 3, 4, 5])  # 绑定到大核

lane_colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
]


def main():
    print(os.getcwd())
    vedioPaths = os.path.join(os.getcwd(), "cut_col_cut_edge","videos")
    videos = get_video_paths(vedioPaths)

    for video in videos:
        # 打开视频对象
        save_path = os.path.join(os.getcwd(), "cut_col_cut_edge","output","{}.mp4".format(Path(video).stem))
        capture = cv2.VideoCapture(video)
        out = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            capture.get(cv2.CAP_PROP_FPS),
            (672, 384),
        )
        if not capture.isOpened():
            print("Open Video Failed")
            return

        with tqdm(
            total=capture.get(cv2.CAP_PROP_FRAME_COUNT)
            - capture.get(cv2.CAP_PROP_POS_FRAMES)
        ) as _tqdm:
            _tqdm.set_description(f"Processing {os.path.basename(video)}")

            # 创建算法对像
            algorithm = Algorithm(
                lane_detector_model=r"D:\nextvpu\multimodel_drive\splitOnnxModel\modelPath\model_sim.onnx",
            )
            # 循环读取视频帧
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break

                # 模拟Sensor输出的YUV格式图片
                bgr = cv2.resize(frame, (672, 384))
                yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
                y, u, v = cv2.split(yuv)
                u = cv2.resize(u, (int(u.shape[1] / 2), int(u.shape[0] / 2)))
                v = cv2.resize(v, (int(v.shape[1] / 2), int(v.shape[0] / 2)))
                uv = np.column_stack((u.flatten(), v.flatten())).reshape(
                    int(y.shape[0] / 2), -1
                )
                yuv_nv12 = np.concatenate((y, uv), axis=0)

                # 算法处理获取结果
                lane_results = algorithm.process(
                    image=yuv_nv12,
                )

                # 可视化结果
                frame = deepcopy(bgr)
                for idx, lane_result in enumerate(lane_results):
                    for pt in lane_result:
                        cv2.circle(
                            frame,
                            (int(pt[0]), int(pt[1])),
                            1,
                            color=lane_colors[idx],
                            thickness=2,
                        )

                # cv2.namedWindow("Debug", cv2.WINDOW_FREERATIO)
                # cv2.imshow("Debug", frame)
                # cv2.waitKey(1)
                out.write(frame)

                _tqdm.update(1)
        capture.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
