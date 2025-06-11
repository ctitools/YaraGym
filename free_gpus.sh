#!/bin/bash
# Script to free up GPU memory by killing all GPU-using processes

echo "Freeing up GPU memory..."

# Kill all axolotl processes
echo "Killing axolotl processes..."
pkill -f "axolotl" 2>/dev/null || true

# Kill all Python processes (be careful with this in production!)
echo "Killing Python processes..."
pkill -f "python" 2>/dev/null || true

# Kill any remaining CUDA processes
echo "Killing CUDA processes..."
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [ ! -z "$pid" ]; then
        echo "Killing PID: $pid"
        kill -9 $pid 2>/dev/null || true
    fi
done

# Wait for processes to terminate
sleep 3

# Force kill any stubborn processes
echo "Force killing any remaining processes..."
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [ ! -z "$pid" ]; then
        echo "Force killing PID: $pid"
        kill -KILL $pid 2>/dev/null || true
    fi
done

# Final wait
sleep 2

# Show GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo ""

# Check if GPUs are free
GPU0_MEM=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
GPU1_MEM=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)

if [ "$GPU0_MEM" -lt 100 ] && [ "$GPU1_MEM" -lt 100 ]; then
    echo "✓ Both GPUs are now free!"
else
    echo "⚠ Warning: Some GPU memory is still in use"
    echo "  GPU 0: ${GPU0_MEM}MiB"
    echo "  GPU 1: ${GPU1_MEM}MiB"
    echo ""
    echo "You may need to run this script again or manually kill processes"
fi