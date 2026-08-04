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
    digital_reference_lead_samples: int | None

    def reset(self) -> None: ...

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        """ref/err: (hop,) float32 → anti-noise (hop,) float32."""
        ...


def checkpoint_digital_reference_lead_samples(state: dict) -> int:
    """체크포인트의 학습 lead를 반환한다 (기존 artifact는 lead=0 호환)."""
    cfg = state.get("cfg", {}) or {}
    if "digital_reference_lead_samples" in cfg:
        return int(cfg["digital_reference_lead_samples"])
    # 개발 중 full cfg를 저장했던 임시 artifact도 읽을 수 있게 한다.
    data_cfg = cfg.get("data", {}) or {}
    return int(data_cfg.get("digital_reference_lead_samples", 0))


def _load_ckpt_model(ckpt_path: str | Path):
    import torch

    from ..models import build_model

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"])
    model.digital_reference_lead_samples = checkpoint_digital_reference_lead_samples(state)
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
        self.digital_reference_lead_samples = int(
            self.model.digital_reference_lead_samples
        )
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
        # key가 없는 기존 ONNX artifact는 기존 정렬인 lead=0으로 호환한다.
        self.digital_reference_lead_samples = int(
            meta.get("digital_reference_lead_samples", 0)
        )
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
        except ImportError as exc:
            raise RuntimeError(
                "tensorrt 파이썬 바인딩이 없습니다. docs/06_deployment_jetson.md 의 "
                "TensorRT 설치 절을 참조하세요."
            ) from exc
        # cuda-python 12 부터 cudart 가 cuda.bindings.runtime 으로 옮겨졌다. 두 배치를
        # 모두 받아준다 — Jetson 이미지마다 버전이 달라 한쪽만 지원하면 배포가 막힌다.
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            try:
                from cuda import cudart
            except ImportError as exc:
                raise RuntimeError(
                    "cuda-python 바인딩이 없습니다 (cuda.bindings.runtime / cuda.cudart "
                    "둘 다 없음). docs/06_deployment_jetson.md 참조."
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
        self.digital_reference_lead_samples = int(
            meta.get("digital_reference_lead_samples", 0)
        )
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

    def __init__(
        self,
        secondary_npz: str,
        fxlms_cfg: dict,
        hop: int = 256,
        handoff_extra_samples: int | None = None,
    ) -> None:
        from ..baselines.fxlms_core import FxLMSController, load_secondary_path

        self.hop = int(hop)
        if self.hop <= 0:
            raise ValueError("FxLMS hop은 양수여야 합니다")
        # RealtimeANC는 입력 블록을 추론 스레드로 넘기고 다음 콜백에서 y를
        # 재생하므로 직접-callback legacy 구현보다 정확히 1 hop이 더 늦다.
        handoff = self.hop if handoff_extra_samples is None else int(handoff_extra_samples)
        if handoff < 0:
            raise ValueError("FxLMS handoff_extra_samples는 0 이상이어야 합니다")
        model = load_secondary_path(secondary_npz)
        self.handoff_extra_samples = handoff
        self.secondary_delay_samples = int(model.delay_samples) + handoff
        self.secondary_total_length = self.secondary_delay_samples + int(model.fir.size)
        self.controller = FxLMSController(
            model.fir,
            secondary_delay_samples=self.secondary_delay_samples,
            control_len=int(fxlms_cfg.get("control_length", 256)),
            mu=float(fxlms_cfg.get("mu", 0.05)),
            leakage=float(fxlms_cfg.get("leakage", 1.0e-6)),
            weight_norm_limit=float(fxlms_cfg.get("weight_norm_limit", 20.0)),
        )
        # 학습 체크포인트가 없는 적응 필터이므로 runtime lead를 별도로 제한하지 않는다.
        self.digital_reference_lead_samples = None
        # ANC OFF 베이스라인 중 가중치가 몰래 누적되지 않도록 fail-closed로 시작한다.
        self.adapt = False
        self.last_adaptation = None

    def reset(self) -> None:
        self.controller.reset(reset_histories=True)
        self.adapt = False
        self.last_adaptation = None

    def set_adapt_enabled(self, enabled: bool) -> None:
        self.adapt = bool(enabled)

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        y = self.controller.generate_block(ref)
        self.last_adaptation = self.controller.adapt_block(err, enabled=self.adapt)
        return y


def secondary_path_npz(runtime_cfg: dict) -> str:
    """S(z) npz 경로 — duct.yaml secondary_path.npz 가 단일 출처 (감사 M9)."""
    from ..config import _resolve_path

    return str(_resolve_path(runtime_cfg["duct"]["secondary_path"]["npz"]))


def build_engine(runtime_cfg: dict) -> InferenceEngine:
    """runtime.yaml 로 엔진 구성. controller=fxlms 면 FxLMSEngine."""
    hop = int(runtime_cfg.get("hop", 256))
    if runtime_cfg.get("controller", "dl") == "fxlms":
        handoff = int(
            runtime_cfg.get("duct", {})
            .get("secondary_path", {})
            .get("handoff_extra_samples", hop)
        )
        if handoff != hop:
            raise ValueError(
                "실시간 FxLMS의 handoff_extra_samples는 실제 1-hop 파이프라인과 "
                f"같아야 합니다: 설정={handoff}, hop={hop}"
            )
        return FxLMSEngine(
            secondary_path_npz(runtime_cfg),
            runtime_cfg.get("fxlms", {}),
            hop=hop,
            handoff_extra_samples=handoff,
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
