#!/bin/bash

# Quick status check for VAE training job
echo "=== VAE Training Status Check ==="
echo "Time: $(date)"
echo

# Job status
echo "1. Job Queue Status:"
squeue -j 1682 2>/dev/null || echo "Job not found in queue (may have completed)"
echo

# Check for log files
echo "2. Log Files:"
if [ -f "logs/vae_training_1682.out" ]; then
    echo "✓ Output log exists ($(wc -l < logs/vae_training_1682.out) lines)"
else
    echo "✗ No output log yet"
fi

if [ -f "logs/vae_training_1682.err" ]; then
    echo "✓ Error log exists ($(wc -l < logs/vae_training_1682.err) lines)"
else
    echo "✗ No error log yet"
fi
echo

# Check directories
echo "3. Output Directories:"
echo "Checkpoints: $(ls -1 vae_checkpoints/ 2>/dev/null | wc -l) files"
echo "Logs: $(ls -1 vae_logs/ 2>/dev/null | wc -l) items"
echo

# Recent activity
echo "4. Recent System Activity:"
echo "Last 5 jobs by user:"
squeue -u $(whoami) -S -i | head -6

echo
echo "=== Monitoring Commands ==="
echo "• Check job: squeue -j 1682"
echo "• View output: tail -f logs/vae_training_1682.out"
echo "• View errors: tail -f logs/vae_training_1682.err"
echo "• Cancel job: scancel 1682"
echo "• Full monitor: ./monitor_training.sh"