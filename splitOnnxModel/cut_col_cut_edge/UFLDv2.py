import os, cv2, shutil, psutil
import onnxruntime as ort
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
import json
import torch

psutil.Process(os.getpid()).cpu_affinity([0, 1, 2, 3, 4, 5])  # 绑定到大核

NUM_OF_GRID = 100
NUM_OF_ROW = 56
NUM_OF_LANE = 2


def softmax(x, axis=0):
    # 减去最大值以提高数值稳定性
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class UFLDv2:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(
            self.model_path,
            providers=[
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        self.input_shape = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name




        self.session_1 = ort.InferenceSession(r"D:\nextvpu\multimodel_drive\splitOnnxModel\modelPath/model_sim_above.onnx")
        self.session_2 = ort.InferenceSession(r"D:\nextvpu\multimodel_drive\splitOnnxModel\modelPath/model_sim_below.onnx")
        # split onnx的输入输出名称  
        # 获取输入输出名称（假设模型只有一个输入输出）
        self.input_name_1 = self.session_1.get_inputs()[0].name
        self.output_name_1 = self.session_1.get_outputs()[0].name
        self.input_name_2 = self.session_2.get_inputs()[0].name


        self.num_row = 56
        self.num_col = 41
        self.row_anchor = np.linspace(160,710, self.num_row)/720
        self.col_anchor = np.linspace(0,1, self.num_col)
    def __call__(
        self,
        img,
        confidence: float = 0.5,
        original_image_width=672,
        original_image_height=384,
    ):
        return self.predict(
            image=img,
            confidence=confidence,
            original_image_width=original_image_width,
            original_image_height=original_image_height,
        )

    def preprocess(self, image):
        # Color YUV_NV12 to RGB
        image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_NV12)
        image = (image.astype(np.float32) / 255.0 - (0.485, 0.456, 0.406)) / (
            0.229,
            0.224,
            0.225,
        )

        return image.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]


    def run_pipeline(self, input_data):
        """
        input_data: 第一个模型的输入数据 (例如图像预处理后的 tensor)
        返回最终第二个模型的输出
        """
        # Step 1: 推理第一个模型
        output_1 = self.session_1.run(None, {self.input_name_1: input_data})[0]  # shape: (1, 8, 12, 21)

        # Step 2: 转置输出以匹配第二个模型输入格式
        transposed_output = np.transpose(output_1, (0, 3, 2, 1))  # -> (1, 21, 12, 8)
        # Step 3: 展平为二维输入，适配第二个模型输入要求
        flattened_input = transposed_output.reshape(1, -1)  # -> (1, 2016)
        # Step 3: 推理第二个模型
        output_final = self.session_2.run(None, {self.input_name_2: flattened_input})

        return output_final

    def predict(
        self,
        image,
        confidence: float = 0.5,
        original_image_width=672,
        original_image_height=384,
    ):
        # Preprocess
        tensor = self.preprocess(image)

        # 双模型推理
        # print("双模型推理")
        # output = self.run_pipeline(tensor)
        # 单模型推理
        print("单模型推理")
        output = self.session.run(None, {self.input_name: tensor})
        loc_row = output[0][:, : NUM_OF_GRID * NUM_OF_ROW * NUM_OF_LANE].reshape(1, NUM_OF_GRID, NUM_OF_ROW, NUM_OF_LANE)
        exist_row = output[0][:, NUM_OF_GRID * NUM_OF_ROW * NUM_OF_LANE :].reshape(1, 2, NUM_OF_ROW, NUM_OF_LANE)

        # Postprocess
        points = self.postprocess_points(
            loc_row,
            exist_row,
            confidence=confidence,
            original_image_width=original_image_width,
            original_image_height=original_image_height,
        )
        return points

    def postprocess_points(
        self,
        loc_row,
        exist_row,
        confidence: float = 0.5,
        row_anchor=np.linspace(160, 710, 56) / 720,
        local_width=1,
        original_image_width=1640,
        original_image_height=590,
    ):
        batch_size, num_grid_row, num_cls_row, num_lane_row = loc_row.shape

        max_indices_row = np.argmax(loc_row, axis=1)
        # n , num_cls, num_lanes
        valid_row = np.argmax(softmax(exist_row, axis=1) > confidence, axis=1)
        # n, num_cls, num_lanes

        lane_idx = [0, 1]

        lanes = []
        for i in lane_idx:
            tmp = []
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = np.array(
                        list(
                            range(
                                max(0, max_indices_row[0, k, i] - local_width),
                                min(
                                    num_grid_row - 1,
                                    max_indices_row[0, k, i] + local_width,
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = np.sum(softmax(loc_row[0, all_ind, k, i], axis=0) * all_ind.astype(np.float32)) + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append([int(out_tmp), int(row_anchor[k] * original_image_height)])
            lanes.append(tmp)

        return lanes
