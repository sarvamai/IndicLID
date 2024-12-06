## Indic-LID Triton Inference

1. Download the checkpoints to `models` folder inside `ai4bharat` folder.
2. Build the image: `docker build -t indiclid_triton .`
3. Start the container: `docker run --shm-size=256m --rm -p 8000:8000 -t indiclid_triton`
4. Check sample outputs: `python triton_repo/client.py`
