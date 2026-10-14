`external_scalar.onnx` is a 172-byte synthetic ONNX IR 8 / opset 18 model:

```text
input x: float[1]
initializer weight: float[1], external_data={location: "weight.bin", offset: "0", length: "4"}
output y: float[1] = Add(x, weight)
```

Tests generate `weight.bin` in temporary directories with distinct values.
There are no trained weights, downloaded assets, or audio recordings here.

Equivalent model construction using the optional Python `onnx` package:

```python
import onnx
from onnx import TensorProto, helper

weight = TensorProto(name="weight", data_type=TensorProto.FLOAT, dims=[1])
weight.data_location = TensorProto.EXTERNAL
for key, value in [("location", "weight.bin"), ("offset", "0"), ("length", "4")]:
    entry = weight.external_data.add()
    entry.key, entry.value = key, value
graph = helper.make_graph(
    [helper.make_node("Add", ["x", "weight"], ["y"])],
    "external_scalar",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    [weight],
)
model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 18)])
model.producer_name = "seyk regression fixture"
with open("external_scalar.onnx", "wb") as stream:
    stream.write(model.SerializeToString())
```
