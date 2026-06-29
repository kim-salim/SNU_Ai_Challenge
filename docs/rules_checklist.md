# Rules Checklist

- [ ] External data is not used
- [ ] External commercial APIs are not used
- [ ] Test images are not manually inspected
- [ ] Test statistics are not used for thresholding or design changes
- [ ] Ensembling is not used
- [ ] Multiple split fine-tuning result combination is not used
- [ ] Model public release date is checked against 2026-05-31
- [ ] Local offline execution is checked
- [ ] RTX 3090 24GB feasibility is checked
- [ ] 24 hour full test inference feasibility is checked
- [ ] Separate `inference.py` path exists
- [ ] Relative paths are used
- [ ] UTF-8 file encoding is used
- [ ] Total code and model artifact size is below 80GB

