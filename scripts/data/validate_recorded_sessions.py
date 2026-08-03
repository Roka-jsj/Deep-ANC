#!/usr/bin/env python3
"""실측 세션 전수 QA (읽기 전용, 오디오 출력 없음).

manifest 경로를 ``read_manifest``로 해석한 뒤 mics/source/session metadata를
블록 단위로 검사하고 Markdown+JSON 리포트를 저장한다. 오류가 하나라도 있으면
비정상 종료한다.

  .venv/bin/python scripts/data/validate_recorded_sessions.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_yaml  # noqa: E402
from deep_anc.data.manifest import VALID_SPLITS, read_manifest  # noqa: E402
from deep_anc.data.recorded_qa import (  # noqa: E402
    failure_report,
    render_recorded_qa_markdown,
    settings_from_data_config,
    validate_recorded_sessions,
)


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _write_reports(report: dict, markdown_path: Path, json_path: Path) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_recorded_qa_markdown(report), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/manifests/recorded_train.jsonl"
    )
    parser.add_argument("--data-config", default="configs/data_sim.yaml")
    parser.add_argument("--out-md", default="data/manifests/recorded_qa.md")
    parser.add_argument("--out-json", default="data/manifests/recorded_qa.json")
    parser.add_argument("--block-frames", type=int, default=262_144)
    parser.add_argument("--clip-threshold", type=float, default=0.99)
    parser.add_argument("--max-clip-ratio", type=float, default=0.005)
    parser.add_argument("--min-mic-rms-dbfs", type=float, default=-80.0)
    parser.add_argument("--min-source-rms-dbfs", type=float, default=-80.0)
    parser.add_argument(
        "--required-splits",
        nargs="*",
        choices=VALID_SPLITS,
        default=list(VALID_SPLITS),
        help="세션이 반드시 하나 이상 있어야 하는 split (빈 목록이면 커버리지 요약만)",
    )
    parser.add_argument(
        "--allow-incomplete-family-coverage",
        action="store_true",
        help=(
            "진단용: source_family가 required split 일부에 없어도 경고로 낮춤 "
            "(기본 G2 게이트는 치명 오류)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = _repo_path(args.manifest)
    data_config_path = _repo_path(args.data_config)
    out_md = _repo_path(args.out_md)
    out_json = _repo_path(args.out_json)

    try:
        data_cfg = load_yaml(data_config_path)
        settings = settings_from_data_config(
            data_cfg,
            block_frames=args.block_frames,
            clip_threshold=args.clip_threshold,
            max_clip_ratio=args.max_clip_ratio,
            min_mic_rms_dbfs=args.min_mic_rms_dbfs,
            min_source_rms_dbfs=args.min_source_rms_dbfs,
            required_splits=args.required_splits,
            allow_incomplete_family_coverage=args.allow_incomplete_family_coverage,
        )
        # 상대 경로 marker는 여기에서 manifest 부모 기준 절대경로로 바뀐다.
        entries = read_manifest(manifest_path)
        report = validate_recorded_sessions(
            entries, settings, manifest_path=str(manifest_path)
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        report = failure_report(str(exc), manifest_path=str(manifest_path))

    _write_reports(report, out_md, out_json)
    verdict = "PASS" if report["ok"] else "FAIL"
    summary = report["summary"]
    print(
        f"[{verdict}] 실측 QA: {summary['valid_sessions']}/{summary['sessions']} 세션, "
        f"{float(summary['duration_s']) / 60.0:.2f}분"
    )
    print(f"Markdown: {out_md}")
    print(f"JSON: {out_json}")
    if report.get("errors"):
        for message in report["errors"]:
            print(f"[오류] {message}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
