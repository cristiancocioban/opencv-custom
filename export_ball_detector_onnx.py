import os
import shutil

import onnx
from onnx import helper, TensorProto
from ultralytics import YOLO

model_name = "./models/ball_detection_v26n_640_07_04.pt"
imgsz = 256


def strip_nms(input_path, output_path):
    """Remove TopK/NMS postprocessing nodes from the ONNX graph.

    The YOLO11 detect head bakes a one-to-one TopK selection into the ONNX
    export.  OpenCV DNN computes those ops incorrectly, so we truncate the
    graph at the Transpose node that produces the raw [1, N, 5] tensor
    ([x1, y1, x2, y2, score] in letterbox coords) and let our C++ code
    pick the best detection directly.
    """
    model = onnx.load(input_path)
    target_output = "/model.23/Transpose_output_0"

    # Check the node exists
    all_outputs = set()
    for n in model.graph.node:
        for o in n.output:
            all_outputs.add(o)

    if target_output not in all_outputs:
        print(f"  WARNING: {target_output} not found — skipping strip")
        return

    # Map every tensor name to the node that produces it
    output_to_node = {}
    for n in model.graph.node:
        for o in n.output:
            output_to_node[o] = n

    # Walk backwards from the target to collect only the needed nodes
    visited = set()
    queue = [target_output]
    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        if name in output_to_node:
            for inp in output_to_node[name].input:
                queue.append(inp)

    needed = {output_to_node[o].name for o in visited if o in output_to_node}
    new_nodes = [n for n in model.graph.node if n.name in needed]

    new_output = helper.make_tensor_value_info(
        target_output, TensorProto.FLOAT, [1, None, 5]
    )
    new_graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        model.graph.input,
        [new_output],
        model.graph.initializer,
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version

    onnx.save(new_model, output_path)

    # Verify
    m = onnx.load(output_path)
    topk_count = sum(1 for n in m.graph.node if n.op_type == "TopK")
    print(
        f"  Saved {output_path}  "
        f"({len(m.graph.node)} nodes, {topk_count} TopK, "
        f"was {len(model.graph.node)} nodes)"
    )


# --- Export ONNX from PyTorch -------------------------------------------
model = YOLO(model_name)
model.export(format="onnx", simplify=True, opset=12, nms=False, imgsz=imgsz)

# Rename to reflect actual imgsz
default_onnx = model_name.replace(".pt", ".onnx")
desired_onnx = model_name.replace("640", str(imgsz)).replace(".pt", ".onnx")
if default_onnx != desired_onnx and os.path.exists(default_onnx):
    shutil.move(default_onnx, desired_onnx)
    print(f"Renamed to {desired_onnx}")

# --- Strip NMS/TopK nodes for OpenCV DNN compatibility ------------------
raw_onnx = desired_onnx.replace(".onnx", "_raw.onnx")
print(f"Stripping NMS nodes: {desired_onnx} -> {raw_onnx}")
strip_nms(desired_onnx, raw_onnx)

print("Done!")
