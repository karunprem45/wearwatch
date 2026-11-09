"""Export the trained forest to ONNX and check ONNX Runtime gives the same probabilities.

Needs skl2onnx (pip install skl2onnx). The parity check is the important part: an
exported model that disagrees with the original is a silent production bug.
"""
import numpy as np

from .features import FEATURES


def to_onnx(model, path="models/model.onnx"):
    from skl2onnx import to_onnx as _to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    onx = _to_onnx(model, initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
                   options={id(model): {"zipmap": False}}, target_opset=17)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return path


def parity(model, session, X):
    """Max absolute difference in failure probability between scikit-learn and ONNX Runtime."""
    p_sk = model.predict_proba(X)[:, 1]
    outs = session.run(None, {"input": X.astype(np.float32)})
    p_ort = outs[1][:, 1]
    return float(np.max(np.abs(p_sk - p_ort)))
