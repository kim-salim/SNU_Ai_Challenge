from PIL import Image

from snu_order.vlm24.image_builder import make_labeled_grid_2x2


def test_grid() -> None:
    frames = [
        Image.new("RGB", (80, 60), color=(255, 0, 0)),
        Image.new("RGB", (60, 80), color=(0, 255, 0)),
        Image.new("RGB", (70, 70), color=(0, 0, 255)),
        Image.new("RGB", (90, 50), color=(255, 255, 0)),
    ]
    grid = make_labeled_grid_2x2(frames, grid_size=256, label_height=32)
    assert grid.mode == "RGB"
    assert grid.size == (256, 256)
    assert grid.getpixel((10, 10)) != (255, 255, 255)
    assert grid.getpixel((138, 10)) != (255, 255, 255)
    assert grid.getpixel((10, 138)) != (255, 255, 255)
    assert grid.getpixel((138, 138)) != (255, 255, 255)
