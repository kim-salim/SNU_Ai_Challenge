from __future__ import annotations

from huggingface_hub import snapshot_download


def main() -> None:
    snapshot_download(
        repo_id="google/siglip2-base-patch16-224",
        local_dir="weights/pretrained/siglip2_base_224",
    )
    print("[OK] downloaded google/siglip2-base-patch16-224")
    print("[OK] local_dir = weights/pretrained/siglip2_base_224")


if __name__ == "__main__":
    main()
