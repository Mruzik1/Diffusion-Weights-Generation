#!/bin/bash

# VAE Training Monitor Script
echo "=== VAE Training Monitor ==="
echo "Job ID: 1682"
echo "Monitoring started at: $(date)"
echo

# Function to check job status
check_job_status() {
    job_status=$(squeue -j 1682 -h -o "%T" 2>/dev/null)
    if [ -z "$job_status" ]; then
        echo "Job completed or not found"
        return 1
    else
        echo "Job status: $job_status"
        return 0
    fi
}

# Function to show recent log output
show_logs() {
    if [ -f "logs/vae_training_1682.out" ]; then
        echo "=== Recent Output ==="
        tail -20 logs/vae_training_1682.out
        echo
    fi
    
    if [ -f "logs/vae_training_1682.err" ]; then
        echo "=== Recent Errors ==="
        tail -10 logs/vae_training_1682.err
        echo
    fi
}

# Main monitoring loop
while true; do
    clear
    echo "=== VAE Training Monitor - $(date) ==="
    echo
    
    # Check job status
    if ! check_job_status; then
        echo "Job finished. Final logs:"
        show_logs
        break
    fi
    
    # Show logs if job is running
    show_logs
    
    # Show queue info
    echo "=== Queue Status ==="
    squeue -u $(whoami)
    echo
    
    # Show GPU usage if job is running
    if [ "$job_status" = "RUNNING" ]; then
        echo "=== GPU Monitoring ==="
        echo "Use 'nvidia-smi' on the compute node to monitor GPU usage"
        echo
    fi
    
    echo "Press Ctrl+C to stop monitoring (job will continue)"
    sleep 30
done