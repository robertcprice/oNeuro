#!/usr/bin/env bash
# =============================================================================
# Vast.ai deployment script for oNeuro dONN GPU experiments
#
# Deploys to an H200/A100/4090 instance and runs language learning,
# DishBrain replication, and Spatial Arena experiments at GPU scale.
#
# Usage:
#   # Step 1: Find cheapest GPU instance
#   bash scripts/vast_deploy.sh search
#
#   # Step 2: Create instance (use offer_id from search)
#   bash scripts/vast_deploy.sh create <offer_id>
#
#   # Step 3: Deploy oNeuro and run experiments
#   bash scripts/vast_deploy.sh deploy <instance_id>
#
#   # Step 4: Run experiments
#   bash scripts/vast_deploy.sh run <id> [scale]                # language learning
#   bash scripts/vast_deploy.sh dishbrain <id> [scale] [flags]  # DishBrain Pong
#   bash scripts/vast_deploy.sh doom <id> [scale] [flags]       # Spatial Arena
#   bash scripts/vast_deploy.sh all <id> [scale]                # everything
#
#   # Step 5: Collect results and destroy
#   bash scripts/vast_deploy.sh results <instance_id>
#   bash scripts/vast_deploy.sh destroy <instance_id>
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_KEY
# =============================================================================

set -euo pipefail

DOCKER_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DISK_SIZE=50  # GB
REPO_URL="https://github.com/bobbyprice/oNeuro.git"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[oNeuro]${NC} $*"; }
warn() { echo -e "${YELLOW}[oNeuro]${NC} $*"; }
err() { echo -e "${RED}[oNeuro]${NC} $*" >&2; }

# =============================================================================
# Commands
# =============================================================================

cmd_search() {
    log "Searching for GPU instances..."
    echo ""
    echo "=== H200 (141GB HBM3e — best for 1M neurons) ==="
    vastai search offers 'gpu_name=H200 num_gpus=1 inet_down>500 reliability>0.98' \
        --order 'dph_total' --limit 5 2>/dev/null || echo "  No H200 found"

    echo ""
    echo "=== A100 80GB (good for 500K neurons) ==="
    vastai search offers 'gpu_name=A100_SXM4 gpu_ram>=80000 num_gpus=1 inet_down>500 reliability>0.98' \
        --order 'dph_total' --limit 5 2>/dev/null || echo "  No A100 found"

    echo ""
    echo "=== RTX 4090 (24GB — good for 100K neurons) ==="
    vastai search offers 'gpu_name=RTX_4090 num_gpus=1 inet_down>200 reliability>0.95' \
        --order 'dph_total' --limit 5 2>/dev/null || echo "  No 4090 found"

    echo ""
    log "Pick an offer_id and run: bash scripts/vast_deploy.sh create <offer_id>"
}

cmd_create() {
    local offer_id="${1:?Usage: vast_deploy.sh create <offer_id>}"
    log "Creating instance from offer ${offer_id}..."

    vastai create instance "${offer_id}" \
        --image "${DOCKER_IMAGE}" \
        --disk "${DISK_SIZE}" \
        --onstart-cmd "apt-get update && apt-get install -y git && pip install numpy"

    log "Instance created. Wait for it to start, then run:"
    log "  bash scripts/vast_deploy.sh deploy <instance_id>"
}

cmd_deploy() {
    local instance_id="${1:?Usage: vast_deploy.sh deploy <instance_id>}"
    log "Deploying oNeuro to instance ${instance_id}..."

    # Get SSH info
    local ssh_url
    ssh_url=$(vastai ssh-url "${instance_id}" 2>/dev/null)
    log "SSH: ${ssh_url}"

    # Deploy via SSH
    local ssh_cmd="ssh -o StrictHostKeyChecking=no ${ssh_url}"

    ${ssh_cmd} bash -s <<'DEPLOY_SCRIPT'
set -euo pipefail

echo "=== Installing dependencies ==="
pip install torch numpy 2>/dev/null

echo "=== Cloning oNeuro ==="
cd /workspace
if [ -d "oNeuro" ]; then
    cd oNeuro && git pull
else
    git clone https://github.com/bobbyprice/oNeuro.git
    cd oNeuro
fi

echo "=== Verifying CUDA ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo "=== Quick test ==="
cd /workspace/oNeuro
PYTHONPATH=src python3 -c "
from oneuro.molecular.cuda_backend import CUDAMolecularBrain, CUDARegionalBrain
import time

brain = CUDAMolecularBrain(1000, device='cuda')
brain.add_random_synapses(5000)

# Warmup
brain.run(10)

# Benchmark
start = time.perf_counter()
brain.run(100)
elapsed = time.perf_counter() - start
print(f'1K neurons, 100 steps: {elapsed*1000:.0f}ms ({elapsed*10:.2f}ms/step)')

# Scale test
for n in [10000, 100000, 1000000]:
    brain = CUDAMolecularBrain(n, device='cuda')
    brain.run(3)
    start = time.perf_counter()
    brain.run(10)
    elapsed = time.perf_counter() - start
    ms_per_step = elapsed * 100
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f'{n:>8d} neurons: {ms_per_step:.1f}ms/step, {mem:.1f}GB VRAM')
    torch.cuda.reset_peak_memory_stats()
"

echo ""
echo "=== Deployment complete ==="
echo "Run experiments with: bash scripts/vast_deploy.sh run ${HOSTNAME} [scale]"
DEPLOY_SCRIPT

    log "Deployment complete!"
}

cmd_run() {
    local instance_id="${1:?Usage: vast_deploy.sh run <instance_id> [scale]}"
    local scale="${2:-100k}"
    log "Running language experiments at ${scale} scale on instance ${instance_id}..."

    local ssh_url
    ssh_url=$(vastai ssh-url "${instance_id}" 2>/dev/null)

    ssh -o StrictHostKeyChecking=no "${ssh_url}" bash -s <<RUNCMD
set -euo pipefail
cd /workspace/oNeuro
PYTHONPATH=src python3 demos/demo_language_cuda.py --scale ${scale} --device cuda 2>&1 | tee /workspace/results_${scale}.txt
echo ""
echo "=== Results saved to /workspace/results_${scale}.txt ==="
RUNCMD

    log "Experiments complete! Fetch results with:"
    log "  bash scripts/vast_deploy.sh results ${instance_id}"
}

cmd_dishbrain() {
    local instance_id="${1:?Usage: vast_deploy.sh dishbrain <instance_id> [scale] [--exp N] [--runs N]}"
    local scale="${2:-medium}"
    shift 2 2>/dev/null || true
    local extra_args="$*"
    log "Running DishBrain experiments at ${scale} scale on instance ${instance_id}..."

    local ssh_url
    ssh_url=$(vastai ssh-url "${instance_id}" 2>/dev/null)

    ssh -o StrictHostKeyChecking=no "${ssh_url}" bash -s <<RUNCMD
set -euo pipefail
cd /workspace/oNeuro
PYTHONPATH=src python3 demos/demo_dishbrain_pong.py \\
    --scale ${scale} --device cuda \\
    --json /workspace/dishbrain_${scale}.json \\
    --gpu-tiers \\
    ${extra_args} \\
    2>&1 | tee /workspace/dishbrain_${scale}.txt
echo ""
echo "=== DishBrain results saved ==="
echo "  Text: /workspace/dishbrain_${scale}.txt"
echo "  JSON: /workspace/dishbrain_${scale}.json"
RUNCMD

    log "DishBrain experiments complete!"
}

cmd_doom() {
    local instance_id="${1:?Usage: vast_deploy.sh doom <instance_id> [scale]}"
    local scale="${2:-medium}"
    shift 2 2>/dev/null || true
    local extra_args="$*"
    log "Running Spatial Arena experiments at ${scale} scale on instance ${instance_id}..."

    local ssh_url
    ssh_url=$(vastai ssh-url "${instance_id}" 2>/dev/null)

    ssh -o StrictHostKeyChecking=no "${ssh_url}" bash -s <<RUNCMD
set -euo pipefail
cd /workspace/oNeuro
PYTHONPATH=src python3 demos/demo_doom_arena.py \\
    --scale ${scale} --device cuda \\
    --json /workspace/doom_${scale}.json \\
    ${extra_args} \\
    2>&1 | tee /workspace/doom_${scale}.txt
echo ""
echo "=== Spatial Arena results saved ==="
echo "  Text: /workspace/doom_${scale}.txt"
echo "  JSON: /workspace/doom_${scale}.json"
RUNCMD

    log "Spatial Arena experiments complete!"
}

cmd_drosophila() {
    local instance_id="${1:?Usage: vast_deploy.sh drosophila <instance_id> [scale] [--exp N]}"
    local scale="${2:-small}"
    shift 2 2>/dev/null || true
    local extra_args="$*"
    log "Running Drosophila ecosystem experiments at ${scale} scale on instance ${instance_id}..."

    local ssh_url
    ssh_url=$(vastai ssh-url "${instance_id}" 2>/dev/null)

    ssh -o StrictHostKeyChecking=no "${ssh_url}" bash -s <<RUNCMD
set -euo pipefail
cd /workspace/oNeuro
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 demos/demo_drosophila_ecosystem.py \\
    --scale ${scale} --device cuda \\
    ${extra_args} \\
    2>&1 | tee /workspace/drosophila_${scale}.txt
echo ""
echo "=== Drosophila results saved ==="
echo "  Text: /workspace/drosophila_${scale}.txt"
RUNCMD

    log "Drosophila experiments complete!"
}

cmd_all() {
    local instance_id="${1:?Usage: vast_deploy.sh all <instance_id> [scale]}"
    local scale="${2:-medium}"
    log "Running ALL experiments at ${scale} scale on instance ${instance_id}..."

    cmd_benchmark "${instance_id}"
    cmd_run "${instance_id}" "${scale}"
    cmd_dishbrain "${instance_id}" "${scale}" --runs 3
    cmd_doom "${instance_id}" "${scale}" --runs 3
    cmd_drosophila "${instance_id}" "${scale}"

    log "All experiments complete! Fetch results with:"
    log "  bash scripts/vast_deploy.sh results ${instance_id}"
}

cmd_results() {
    local instance_id="${1:?Usage: vast_deploy.sh results <instance_id>}"
    log "Fetching results from instance ${instance_id}..."

    local ssh_url
    ssh_url=$(vastai ssh-url "${instance_id}" 2>/dev/null)

    mkdir -p results
    scp -o StrictHostKeyChecking=no "${ssh_url}:/workspace/results_*.txt" results/ 2>/dev/null || true
    scp -o StrictHostKeyChecking=no "${ssh_url}:/workspace/dishbrain_*.txt" results/ 2>/dev/null || true
    scp -o StrictHostKeyChecking=no "${ssh_url}:/workspace/dishbrain_*.json" results/ 2>/dev/null || true
    scp -o StrictHostKeyChecking=no "${ssh_url}:/workspace/doom_*.txt" results/ 2>/dev/null || true
    scp -o StrictHostKeyChecking=no "${ssh_url}:/workspace/doom_*.json" results/ 2>/dev/null || true
    scp -o StrictHostKeyChecking=no "${ssh_url}:/workspace/drosophila_*.txt" results/ 2>/dev/null || true
    scp -o StrictHostKeyChecking=no "${ssh_url}:/workspace/benchmark.txt" results/ 2>/dev/null || true

    local n_files
    n_files=$(ls results/ 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_files" -eq 0 ]; then
        warn "No result files found. Run experiments first."
    else
        log "Downloaded ${n_files} files to results/"
        ls -la results/
    fi
}

cmd_destroy() {
    local instance_id="${1:?Usage: vast_deploy.sh destroy <instance_id>}"
    warn "Destroying instance ${instance_id}..."
    vastai destroy instance "${instance_id}"
    log "Instance destroyed."
}

cmd_benchmark() {
    local instance_id="${1:?Usage: vast_deploy.sh benchmark <instance_id>}"
    log "Running scale benchmark on instance ${instance_id}..."

    local ssh_url
    ssh_url=$(vastai ssh-url "${instance_id}" 2>/dev/null)

    ssh -o StrictHostKeyChecking=no "${ssh_url}" bash -s <<'BENCHCMD'
set -euo pipefail
cd /workspace/oNeuro
PYTHONPATH=src python3 -c "
import torch, time
from oneuro.molecular.cuda_backend import CUDAMolecularBrain, CUDARegionalBrain

device = 'cuda'
print('=' * 70)
print(f'GPU: {torch.cuda.get_device_name()}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('=' * 70)
print()

# Raw neuron scaling
print('--- Raw Neuron Scaling (no synapses) ---')
for n in [1000, 10000, 50000, 100000, 500000, 1000000, 3000000]:
    try:
        torch.cuda.reset_peak_memory_stats()
        b = CUDAMolecularBrain(n, device=device)
        b.run(5)  # warmup
        start = time.perf_counter()
        b.run(20)
        elapsed = time.perf_counter() - start
        ms = elapsed / 20 * 1000
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f'  {n:>8,d} neurons | {ms:8.1f} ms/step | {mem:6.1f} GB | {n/ms:.0f} neurons/ms')
        del b
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  {n:>8,d} neurons | OOM ({e})')
        break

print()
print('--- Regional Brain Scaling ---')
for n_cols, label in [(10, 'mega'), (50, '10K'), (100, '50K'), (500, '100K'), (1000, '250K')]:
    try:
        torch.cuda.reset_peak_memory_stats()
        rb = CUDARegionalBrain._build(n_columns=n_cols, n_per_layer=20, device=device, seed=42)
        n, ns = rb.n_neurons, rb.n_synapses
        for s in range(10):
            if s % 2 == 0: rb.stimulate_thalamus(30.0)
            rb.step()
        start = time.perf_counter()
        for s in range(50):
            if s % 4 == 0: rb.stimulate_thalamus(20.0)
            rb.step()
        elapsed = time.perf_counter() - start
        ms = elapsed / 50 * 1000
        mem = torch.cuda.max_memory_allocated() / 1e9
        spk = sum(rb.brain.get_spike_counts())
        print(f'  {label:>6s} ({n:>8,d} neurons, {ns:>10,d} syn) | {ms:8.1f} ms/step | {mem:6.1f} GB | {spk:>8,d} spikes')
        del rb
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  {label:>6s} | OOM ({e})')
        break

print()
print('Done.')
" 2>&1 | tee /workspace/benchmark.txt
BENCHCMD

    log "Benchmark complete!"
}

# =============================================================================
# Main
# =============================================================================

case "${1:-help}" in
    search)    cmd_search ;;
    create)    cmd_create "${2:-}" ;;
    deploy)    cmd_deploy "${2:-}" ;;
    run)       cmd_run "${2:-}" "${3:-100k}" ;;
    dishbrain) shift; cmd_dishbrain "$@" ;;
    doom)      shift; cmd_doom "$@" ;;
    drosophila) shift; cmd_drosophila "$@" ;;
    all)       cmd_all "${2:-}" "${3:-medium}" ;;
    results)   cmd_results "${2:-}" ;;
    destroy)   cmd_destroy "${2:-}" ;;
    benchmark) cmd_benchmark "${2:-}" ;;
    *)
        echo "oNeuro Vast.ai Deployment — dONN GPU Experiments"
        echo ""
        echo "Usage: bash scripts/vast_deploy.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  search                          Search for GPU instances"
        echo "  create <offer_id>               Create a new instance"
        echo "  deploy <id>                     Deploy oNeuro to instance"
        echo "  run <id> [scale]                Run language experiments (mega/100k/1m)"
        echo "  dishbrain <id> [scale] [flags]  Run DishBrain Pong experiments"
        echo "  doom <id> [scale] [flags]       Run Spatial Arena experiments"
        echo "  drosophila <id> [scale] [flags] Run Drosophila ecosystem experiments"
        echo "  all <id> [scale]                Run benchmark + language + dishbrain + doom + drosophila"
        echo "  benchmark <id>                  Run scaling benchmark"
        echo "  results <id>                    Download results (txt + json)"
        echo "  destroy <id>                    Destroy instance"
        echo ""
        echo "Scales: small, medium, large, mega, 100k, 1m"
        echo ""
        echo "Examples:"
        echo "  bash scripts/vast_deploy.sh dishbrain 12345 medium --runs 5"
        echo "  bash scripts/vast_deploy.sh doom 12345 large"
        echo "  bash scripts/vast_deploy.sh all 12345 medium"
        ;;
esac
