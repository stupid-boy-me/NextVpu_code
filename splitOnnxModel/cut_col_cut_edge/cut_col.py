import onnx
import onnx_graphsurgeon as gs
import numpy as np
from pathlib import Path

model_path = "/algdata01/yiguo.huang/project_code/UFLDv2/convertModel/model/model_best-sim.onnx"
graph = gs.import_onnx(onnx.load(model_path))
graph.outputs.clear()
nodes_to_remove = []
nodes_to_add = []
for node in graph.nodes:
    print(node.name)
for node in graph.nodes:
    if node.name in [
        "/Reshape_1",
        "/Reshape_2",
        "/Reshape_3",
        "/Reshape_4",
        "/Slice",
        "/Slice_1",
        "/Slice_2",
        "/Slice_3",
    ]:
        nodes_to_remove.append(node)
    elif node.name == "/cls/cls.3/Gemm":
        nodes_to_remove.append(node)

        weights = node.inputs[1].values
        weights = np.concatenate((weights[0:22400, :], weights[38800:39248, :]), axis=0)
        weights = gs.Constant(node.inputs[1].name, values=weights)
        
        
        bias = node.inputs[2].values
        bias = np.concatenate((bias[0:22400], bias[38800:39248]), axis=0)
        bias = gs.Constant(node.inputs[2].name, values=bias)

        
        output = gs.Variable("lanes", dtype=np.float32, shape=(1, 102 * 56 * 4))
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
onnx.save(model, f"{Path(model_path).stem}_cut_col.onnx")
