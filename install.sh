cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
rm flash_attn-*
cd ..

pip install decord
pip install -U swanlab
pip install "swanlab[dashboard]"
pip install mathruler