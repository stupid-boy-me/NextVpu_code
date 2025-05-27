import onnx
import onnx_graphsurgeon as gs
import numpy as np
from pathlib import Path

model_path = "/algdata01/yiguo.huang/project_code/UFLDv2/convertModel/model/model_best-sim_cut_col.onnx"
graph = gs.import_onnx(onnx.load(model_path))
graph.outputs.clear()
nodes_to_remove = []
nodes_to_add = []
for node in graph.nodes:
    if node.name == "/cls/cls.3/Gemm":
        nodes_to_remove.append(node)

        weights = node.inputs[1].values
        loc_weight = (
            weights[:22400, :]
            .reshape(1, 100, 56, 4, -1)[:, :, :, [1, 2], :]
            .reshape(-1, 480)
        )
        ext_weight = (
            weights[22400:, :]
            .reshape(1, 2, 56, 4, -1)[:, :, :, [1, 2], :]
            .reshape(-1, 480)
        )
        weights = np.concatenate((loc_weight, ext_weight), axis=0)
        weights = gs.Constant(node.inputs[1].name, values=weights)
        bias = node.inputs[2].values
        loc_bias = bias[:22400].reshape(1, 100, 56, 4)[:, :, :, [1, 2]].flatten()
        ext_bias = bias[22400:].reshape(1, 2, 56, 4)[:, :, :, [1, 2]].flatten()
        bias = np.concatenate((loc_bias, ext_bias), axis=0)
        bias = gs.Constant(node.inputs[2].name, values=bias)
        output = gs.Variable("lanes", dtype=np.float32, shape=(1, 102 * 56 * 2))
        node = gs.Node(
            node.op,
            node.name,
            node.attrs,
            inputs=[node.inputs[0], weights, bias],
            outputs=[output],
        )
        nodes_to_add.append(node)

for node in nodes_to_remove:
    graph.nodes.remove(node)

for node in nodes_to_add:
    graph.nodes.append(node)
    graph.outputs.append(node.outputs[0])

graph.cleanup().toposort()
model = onnx.version_converter.convert_version(gs.export_onnx(graph), 11)
model.ir_version = 6
onnx.save(model, f"{Path(model_path).stem}_cut_edge.onnx")
