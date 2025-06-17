#!/bin/bash
# Start GRPO training with vLLM acceleration

echo "Starting YARA GRPO training with 500 steps..."

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "axolotl" 2>/dev/null || true
sleep 2

# Check GPU memory
echo "Current GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used --format=csv

# Activate environment
source grpo_env/bin/activate

# Start vLLM server on GPU 1
echo "Starting vLLM server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup axolotl vllm-serve orion_yara.yaml > vllm_server.log 2>&1 &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for vLLM to start
echo "Waiting for vLLM server to start..."
sleep 200

# Check if vLLM is running
if curl -s http://localhost:8000/health/ > /dev/null; then
    echo "vLLM server is ready!"
else
    echo "ERROR: vLLM server failed to start. Check vllm_server.log"
    exit 1
fi

# Start training on GPU 0
echo "Starting training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup axolotl train orion_yara.yaml --num-processes 1 > training.log 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

echo ""
echo "Training started successfully!"
echo "Monitor progress with:"
echo "  - tail -f training.log"
echo "  - tail -f vllm_server.log"
echo "  - W&B: check your wandb project"
echo ""
echo "To stop training: kill $TRAIN_PID $VLLM_PID"
