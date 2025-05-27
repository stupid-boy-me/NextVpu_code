# 这是实现保留row车道线的验证及转换脚本的逻辑:
1.获取到了训练好的int8的模型
模型地址:
并且将模型更名为model_sim.onnx
2.进行模型切分
切分工具以及切分指令

# 切分reshape之前
splitOnnxModel.exe --modelName model_sim --modelPath D:\nextvpu\multimodel_drive\splitOnnxModel\modelPath --modelInputName images --bottomName /model/features/features.0/features.0.0/_input_quantizer/QuantizeLinear --topName /pool/Conv
更改名字    model_sim_above_weight.onnx
# 切分reshape之后
splitOnnxModel.exe --modelName model_sim --modelPath D:\nextvpu\multimodel_drive\splitOnnxModel\modelPath --modelInputName /Reshape_output_0 --bottomName /cls/cls.1/_input_quantizer/QuantizeLinear --topName /cls/cls.3/Gemm_row --inputShapes 1,2016
更改名字    model_sim_below.onnx

# 对 model_sim_below.onnx 进行权重维度变换 并且保存onnx为 model_sim_below_weight.onnx
实现代码:/algdata01/yiguo.huang/project_code/UFLDv2/liuxiao/quant_fdmobile/split_onnx/split_onnx_split_gemm.py  




