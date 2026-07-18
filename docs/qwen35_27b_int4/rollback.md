# Rollback

27B 실험이 어떤 gate에서 실패해도 production fallback은 변경하지 않는다.

- Branch: `shihoon/qwen35-stage-e1-canonical-84p2`
- HEAD: `003c035c5335355f03a45cbf01af45866c1af85e`
- TREE: `4cff067f0bc39ddd06cf41b68e5c566deb9dbf62`
- Artifact: `model_artifacts/qwen35_stage_e1_canonical_84p2`

Rollback은 새 27B worktree의 실행을 중지하고 위 immutable release를 계속 사용하는 것이다. 보호 worktree에서 reset, checkout, clean, edit를 수행하지 않는다. 27B 파일을 Champion artifact로 복사하거나 overwrite하지 않는다.

