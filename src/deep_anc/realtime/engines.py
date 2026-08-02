"""추론 엔진 — 공통 인터페이스 step(ref, err) → anti-noise.

마이그레이션 경로 (docs/06): torch(개발) → ort(등가성 검증) → trt(배포).
모든 엔진은 내부 상태를 보관하며 reset() 으로 제로 초기화한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class InferenceEngine(Protocol):
    hop: int

    def reset(self) -> None: ...

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        """ref/err: (hop,) float32 → anti-noise (hop,) float32."""
        ...


def _load_ckpt_model(ckpt_path: str | Path):
    import torch

    from ..models import build_model

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"])
    return model.eval()


class TorchEngine:
    """PyTorch eager 스트리밍 (개발/디버깅용 — 커널 런치 오버헤드 큼)."""

    def __init__(self, ckpt: str, hop: int = 256, device: str | None = None) -> None:
        import torch

        self.hop = int(hop)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = _load_ckpt_model(ckpt).to(device)
        self._torch = torch
        self.reset()

    def reset(self) -> None:
        self.states = self.model.init_states(1, self.device)

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = np.stack([ref, err]).astype(np.float32)[None]      # [1,2,hop]
        with torch.no_grad():
            xt = torch.from_numpy(x).to(self.device)
            y, self.states = self.model.streaming_step(xt, self.states)
        return y.squeeze().float().cpu().numpy()


class OrtEngine:
    """ONNX Runtime CPU — export 정합성 검증·CPU 폴백용."""

    def __init__(self, onnx_path: str, hop: int = 256) -> None:
        import json

        import onnxruntime as ort

        self.hop = int(hop)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2       # Tegra affinity 크래시 회피 (명시 지정)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        meta_path = Path(onnx_path).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.state_names: list[str] = meta["state_names"]
        self._init_shapes = {
            i.name: (i.shape, np.float32) for i in self.sess.get_inputs() if i.name != "x"
        }
        self.reset()

    def reset(self) -> None:
        self.states = {
            name: np.zeros(shape, dtype=dtype)
            for name, (shape, dtype) in self._init_shapes.items()
        }
        # attention mask 상태는 -1e4 초기화 (빈 슬롯 무효화)
        for name in self.states:
            if name.endswith("_attn_m"):
                self.states[name][:] = -1.0e4

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        x = np.stack([ref, err]).astype(np.float32)[None]
        feeds = {"x": x}
        feeds.update(self.states)
        outs = self.sess.run(None, feeds)
        y = outs[0].reshape(-1)
        for name, val in zip(self.state_names, outs[1:]):
            self.states[name] = val
        return y.astype(np.float32)


class TrtEngine:
    """TensorRT 10.x FP16 엔진 — 상태 핑퐁 + execute_async_v3 (배포 경로).

    필요: tensorrt 파이썬 바인딩 + cuda-python. 엔진 빌드는 scripts/export/build_trt.sh.
    """

    def __init__(self, plan: str, onnx_meta: str | None = None, hop: int = 256) -> None:
        import json

        try:
            import tensorrt as trt
            from cuda import cudart
        except ImportError as exc:
            raise RuntimeError(
                "tensorrt/cuda-python 바인딩이 없습니다. docs/06_deployment_jetson.md 의 "
                "TensorRT 설치 절을 참조하세요 (시스템 변경이 필요해 기본 미설치)."
            ) from exc

        self.hop = int(hop)
        self._trt = trt
        self._cudart = cudart
        logger = trt.Logger(trt.Logger.WARNING)
        with open(plan, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        meta_path = Path(onnx_meta) if onnx_meta else Path(plan).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.state_names: list[str] = meta["state_names"]

        err_code, self.stream = cudart.cudaStreamCreate()
        assert err_code == cudart.cudaError_t.cudaSuccess

        # 텐서별 호스트/디바이스 버퍼. 상태는 A/B 핑퐁.
        self.host: dict[str, np.ndarray] = {}
        self.dev: dict[str, int] = {}
        self.state_dev: dict[str, tuple[int, int]] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape)) * 4
            base = name[:-4] if name.endswith("_out") else name
            if base in self.state_names:
                if base not in self.state_dev:
                    a = cudart.cudaMalloc(size)[1]
                    b = cudart.cudaMalloc(size)[1]
                    self.state_dev[base] = (a, b)
                    self.host[base] = np.zeros(shape, dtype=np.float32)
            else:
                self.dev[name] = cudart.cudaMalloc(size)[1]
                self.host[name] = np.zeros(shape, dtype=np.float32)
        self._cur = 0
        self.reset()

    def reset(self) -> None:
        cudart = self._cudart
        for name, (a, b) in self.state_dev.items():
            init = np.zeros_like(self.host[name])
            if name.endswith("_attn_m"):
                init[:] = -1.0e4
            for ptr in (a, b):
                cudart.cudaMemcpy(
                    ptr, init.ctypes.data, init.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        cudart = self._cudart
        x = np.ascontiguousarray(np.stack([ref, err]).astype(np.float32)[None])
        cudart.cudaMemcpyAsync(
            self.dev["x"], x.ctypes.data, x.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        )
        self.context.set_tensor_address("x", self.dev["x"])
        self.context.set_tensor_address("y", self.dev["y"])
        cur = self._cur
        for name in self.state_names:
            a, b = self.state_dev[name]
            self.context.set_tensor_address(name, a if cur == 0 else b)
            self.context.set_tensor_address(f"{name}_out", b if cur == 0 else a)
        self.context.execute_async_v3(self.stream)
        y = self.host["y"]
        cudart.cudaMemcpyAsync(
            y.ctypes.data, self.dev["y"], y.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
        )
        cudart.cudaStreamSynchronize(self.stream)
        self._cur ^= 1
        return y.reshape(-1).copy()


class FxLMSEngine:
    """FxLMS 폴백/베이스라인 — anc_project 검증 구현 사용."""

    def __init__(self, secondary_npz: str, fxlms_cfg: dict, hop: int = 256) -> None:
        from ..baselines.fxlms_core import FxLMSController, load_secondary_path

        self.hop = int(hop)
        model = load_secondary_path(secondary_npz)
        self.controller = FxLMSController(
            model.fir,
            secondary_delay_samples=model.delay_samples,
            control_len=int(fxlms_cfg.get("control_length", 256)),
            mu=float(fxlms_cfg.get("mu", 0.05)),
            leakage=float(fxlms_cfg.get("leakage", 1.0e-6)),
            weight_norm_limit=float(fxlms_cfg.get("weight_norm_limit", 20.0)),
        )
        self.adapt = True

    def reset(self) -> None:
        self.controller.reset(reset_histories=True)

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        y = self.controller.generate_block(ref)
        self.controller.adapt_block(err, enabled=self.adapt)
        return y


def build_engine(runtime_cfg: dict) -> InferenceEngine:
    """runtime.yaml 로 엔진 구성. controller=fxlms 면 FxLMSEngine."""
    hop = int(runtime_cfg.get("hop", 256))
    if runtime_cfg.get("controller", "dl") == "fxlms":
        return FxLMSEngine(
            runtime_cfg["secondary_path"], runtime_cfg.get("fxlms", {}), hop=hop
        )
    eng = runtime_cfg.get("engine", {})
    kind = str(eng.get("type", "torch"))
    if kind == "torch":
        return TorchEngine(eng["ckpt"], hop=hop)
    if kind == "ort":
        return OrtEngine(eng["onnx"], hop=hop)
    if kind == "trt":
        return TrtEngine(eng["plan"], eng.get("onnx_meta"), hop=hop)
    raise ValueError(f"알 수 없는 엔진: {kind}")
