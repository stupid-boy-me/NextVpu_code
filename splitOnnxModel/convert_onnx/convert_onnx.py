import onnx
import numpy as np

# 模型路径
model_path = r"D:\nextvpu\multimodel_drive\splitOnnxModel\modelPath/model_sim_below_old.onnx"
output_model_path = r"D:\nextvpu\multimodel_drive\splitOnnxModel\modelPath/model_sim_below.onnx"


# DequantizeLinear 节点名称
target_node_name = "/cls/cls.1/_weight_quantizer/DequantizeLinear"

# 加载模型
model = onnx.load(model_path)

# 查找 DequantizeLinear 节点
dq_node = None
for node in model.graph.node:
    if node.op_type == "DequantizeLinear" and node.name == target_node_name:
        dq_node = node
        break

if dq_node is None:
    raise ValueError(f"Node '{target_node_name}' not found in the model.")

# 获取第一个输入名（通常是量化数据）
x_input_name = dq_node.input[0]

# 查找 initializer（即 x 是否为常量）
initializer = None
for init in model.graph.initializer:
    if init.name == x_input_name:
        initializer = init
        break

if initializer is None:
    raise ValueError(f"Input '{x_input_name}' of DequantizeLinear is not a constant initializer.")

# 将 initializer 转换为 numpy array（假设数据类型为 int8）
raw_data = np.frombuffer(initializer.raw_data, dtype=np.int8)

# Step 1: reshape to [480, 12, 21, 8]
try:
    reshaped = raw_data.reshape(480, 8, 12, 21)
except ValueError as e:
    raise ValueError("Reshape failed (shape [480, 12, 21, 8]):", e)

# Step 2: transpose to [480, 21, 12, 8]
transposed = reshaped.transpose(0, 3, 2, 1)  # perm=[0,2,1,3]

# Step 3: reshape back to [480, 2016]
final_shape = transposed.reshape(480, -1)  # 480 x 2016

# 更新 raw_data（直接替换 bytes 数据）
initializer.raw_data = final_shape.astype(np.int8).tobytes()

# 验证模型合法性
onnx.checker.check_model(model)

# 保存修改后的模型
onnx.save(model, output_model_path)

print(f"Modified model saved to {output_model_path}")