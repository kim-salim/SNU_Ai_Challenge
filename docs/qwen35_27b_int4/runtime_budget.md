# Runtime Budget

규정 상한은 RTX 3090 24GB, 24시간, package 80GB다. 내부 gate는 더 보수적으로 적용한다.

- Peak allocated/reserved VRAM 목표: 23.0 GiB 이하
- Official Test 전체 추론 환산 목표: 18시간 이하
- Package 목표: 75 GiB 이하
- OOM: 0
- CPU/disk offload: 금지

`benchmark_3090.sh`는 GPU 이름에 RTX 3090이 없으면 PASS하지 않는다. Synthetic image와 canonical prompt로 1-frame hidden extraction, 4-frame Stage/Pair/24-way forward, 반복 prediction hash, load/sample time, peak VRAM을 기록한다.

현재 로컬 호스트는 NVIDIA 드라이버가 응답하지 않으므로 3090 budget은 미검증 상태다. CPU offload나 BF16 fallback 결과로 이를 대체하지 않는다.

