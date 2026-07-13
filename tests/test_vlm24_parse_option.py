from snu_order.vlm24.qwen25_adapter import Qwen25VLAdapter


def test_parse() -> None:
    assert Qwen25VLAdapter.parse_option("A") == "A"
    assert Qwen25VLAdapter.parse_option("Answer: C") == "C"
    assert Qwen25VLAdapter.parse_option("The answer is X.") == "X"
    assert Qwen25VLAdapter.parse_option("no valid option here") is None
