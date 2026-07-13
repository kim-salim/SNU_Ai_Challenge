from snu_order.vlm24.candidates import build_24_candidates


def test_candidates() -> None:
    candidates = build_24_candidates()
    assert len(candidates) == 24
    assert [candidate["label"] for candidate in candidates] == list("ABCDEFGHIJKLMNOPQRSTUVWX")
    assert len({candidate["order"] for candidate in candidates}) == 24
    assert candidates[0]["order"] == (0, 1, 2, 3)
    assert candidates[-1]["order"] == (3, 2, 1, 0)
    assert candidates[0]["text"] == "F1 F2 F3 F4"
