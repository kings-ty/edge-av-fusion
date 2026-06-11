"""TensorRT 8.5 runtime wrapper with torch-tensor I/O bindings.

Instead of pycuda-managed buffers we bind torch CUDA tensors directly
(`tensor.data_ptr()` → execute_async_v2). Two wins on Jetson:
- the log-mel patch from MelPatchExtractor is *already* a CUDA tensor, so the
  classifier input never touches host memory at all (true zero-copy hand-off);
- one allocator (torch's caching allocator) owns all GPU memory — no
  pycuda/torch pool fragmentation fights in the shared LPDDR4x pool.

Engines are built offline by tools/build_trt_engine.py and are specific to
this GPU + TRT version (never commit .plan files).
"""
import logging
from typing import Dict, List

import numpy as np
import tensorrt as trt
import torch

log = logging.getLogger(__name__)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
}


class TrtEngine:
    def __init__(self, engine_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError("TrtEngine requires CUDA")
        self._logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._logger, "")
        with open(engine_path, "rb") as fh, trt.Runtime(self._logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError("failed to deserialize %s (TRT version/GPU mismatch? "
                               "rebuild with tools/build_trt_engine.py)" % engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # pre-allocate output tensors; input binding is set per-call
        self._bindings: List[int] = [0] * self.engine.num_bindings
        self.outputs: Dict[str, torch.Tensor] = {}
        self.input_name = None
        self.input_shape = None
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = _TRT_TO_TORCH[self.engine.get_binding_dtype(i)]
            if self.engine.binding_is_input(i):
                self.input_name, self.input_shape = name, shape
                self._input_index = i
                self._input_dtype = dtype
            else:
                t = torch.empty(shape, dtype=dtype, device="cuda")
                self.outputs[name] = t
                self._bindings[i] = t.data_ptr()
        log.info("TRT engine %s: %s%s -> %s", engine_path, self.input_name,
                 self.input_shape, {k: tuple(v.shape) for k, v in self.outputs.items()})

    def infer(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: CUDA tensor matching the engine's static input shape.
        Synchronous: returns after the GPU finished (correct latency semantics
        for the benchmark — an async launch is not a completed inference)."""
        if tuple(x.shape) != self.input_shape:
            raise ValueError("input %s != engine %s" % (tuple(x.shape), self.input_shape))
        x = x.to(dtype=self._input_dtype, device="cuda").contiguous()
        self._bindings[self._input_index] = x.data_ptr()
        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v2(
                bindings=self._bindings, stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execution failed")
        self.stream.synchronize()
        return self.outputs

    def infer_numpy(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        out = self.infer(torch.as_tensor(x))
        return {k: v.float().cpu().numpy() for k, v in out.items()}
