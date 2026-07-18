# Qwen3.5-27B INT4 Champion Port v1

이 디렉터리는 `qwen35_stage_e1_canonical_84p2`의 의사결정 경로를 보존하면서 언어 backbone만 Qwen3.5-27B로 포트하는 실행 계약을 설명한다. 보호 Champion worktree와 artifact는 읽기 전용이며 9B adapter와 calibration 값은 27B에 이식하지 않는다.

## 데이터 계약

새 27B 학습 config는 `full_train_90_10_v1`을 사용한다.

- 원본: 9,535개
- Train: 8,581개 (`89.9948%`)
- Valid: 954개 (`10.0052%`)
- Train/Valid ID 중복: 0
- Train SHA-256: `caf5bb6a840b97cf8520cded4aceb1217c21881531501bf50a0c6a2bb4361284`
- Valid SHA-256: `d8ba4e72542c7fa1e9129f90f1e064d6a4161e52941065a20b4270134be451ad`

보호 Champion의 원래 `ab_v1` split은 수정하지 않는다. 위 90/10 계약은 새 config와 launcher에만 적용된다.

## 실행 순서

```bash
export QWEN35_27B_BASE_PATH=/absolute/path/to/Qwen3.5-27B
export QWEN35_27B_REVISION=<verified-snapshot-commit>
export PYTHON_BIN=/path/to/python

scripts/qwen35_27b_int4/preflight.sh
scripts/qwen35_27b_int4/audit_model.sh
scripts/qwen35_27b_int4/smoke_synthetic.sh
scripts/qwen35_27b_int4/benchmark_3090.sh
scripts/qwen35_27b_int4/extract_frozen_features.sh
scripts/qwen35_27b_int4/train_frozen_probe.sh
```

위 gate가 모두 통과한 뒤에만 장시간 학습 명령을 사람이 실행한다.

```bash
scripts/qwen35_27b_int4/train_qlora.sh
```

이번 구현 작업은 이 full-data 명령을 자동 실행하지 않는다.

## 로컬 모델이 없을 때

자동 다운로드는 금지한다. 공식 repository를 확인한 뒤 다음 target에 내려받는다.

```bash
hf download Qwen/Qwen3.5-27B \
  --local-dir /home/a/snu-ai-frame-ordering/models/Qwen3.5-27B
```

다운로드 후 `config.json`, module count, snapshot/revision을 `audit_model.sh`로 먼저 봉인한다. 관측 구조가 `5120/64/16/48`과 다르면 학습하지 않는다.

## 현재 머신 상태

2026-07-19 감사 시점에는 로컬 27B snapshot이 없고 `nvidia-smi`가 드라이버와 통신하지 못했다. 따라서 source/synthetic port와 Champion head migration은 검증할 수 있지만 27B NF4 GPU load 및 3090 예산 PASS는 아직 주장할 수 없다.
