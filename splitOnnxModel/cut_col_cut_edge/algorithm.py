import sys
import os
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cut_col_cut_edge.UFLDv2 import *
import numpy as np

class Algorithm:
    def __init__(
        self,
        lane_detector_model: str,
        lane_detector_confidence: float = 0.1,
    ) -> None:
        self.lane_detector = UFLDv2(lane_detector_model)
        self.lane_detector_confidence = lane_detector_confidence

    def __del__(self):
        pass

    def process(
        self,
        image: np.ndarray,
    ):  # Input Image Format: 384(H)*672(W), YUV_420SP_NV12
        H, W = int(image.shape[0] * 2 / 3), image.shape[1]

        lane_results = self.lane_detector.predict(
            image=image,
            confidence=self.lane_detector_confidence,
            original_image_width=W,
            original_image_height=H,
        )

        # 输出结果
        return lane_results
