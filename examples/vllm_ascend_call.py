"""Example model layer calling an independently packaged Ascend operator.

Import this model module in every execution worker during model loading,
before compilation or graph capture. The ops_repo import initializes its
AscendC runtime internally; it does not select the worker's device. This is
an example call site, not an automatically discovered vLLM plugin.
"""

import torch

from ops_repo import add


class OpsRepoAddLayer(torch.nn.Module):
    """Example call site for a deliberately selected FP32 addition operation.

    Both tensors must be nonempty, contiguous, FP32, on the same NPU, and have
    the same shape. This sample implements inference only. Importing the
    module does not inject this layer into any model.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return add(x, y, backend="ascendc")
