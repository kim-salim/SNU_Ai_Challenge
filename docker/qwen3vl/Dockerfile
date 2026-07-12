FROM dl-server:cuda13

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf-cache \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      pytest \
      qwen-vl-utils \
      accelerate \
      peft \
      bitsandbytes \
      safetensors \
      hf_xet && \
    chmod a+rx /home/shpark && \
    chmod -R a+rX /home/shpark/venv

CMD ["bash"]
