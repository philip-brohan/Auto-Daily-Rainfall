#!/bin/bash
# Submit preprocessing jobs to cluster using sbatch.
#
# This script divides the image dataset across multiple cluster jobs,
# each running with parallelization.
#
# Usage:
#   bash scripts/submit_preprocess_jobs.sh SOURCE FILTERED IRREGULAR \\
#       [--num-jobs 6] [--workers-per-job 8] [--time 6:00:00] \\
#       [--partition GPU] [--dry-run]
#
# Example:
#   bash scripts/submit_preprocess_jobs.sh \\
#       /data/scratch/philip.brohan/documents/Daily_Rainfall_UK/jpgs_25pc \\
#       /data/scratch/philip.brohan/documents/Daily_Rainfall_UK/jpgs_25pc_filtered \\
#       /data/scratch/philip.brohan/documents/Daily_Rainfall_UK/jpgs_25pc_irregular \\
#       --num-jobs 6 --workers-per-job 8 --time 6:00:00

set -e

# Defaults
NUM_JOBS=6
WORKERS_PER_JOB=8
TIME_LIMIT="6:00:00"
MEM_PER_WORKER="2GB"
DRY_RUN=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 SOURCE FILTERED IRREGULAR [options]"
    echo "  --num-jobs N              Number of cluster jobs (default: 6)"
    echo "  --workers-per-job N       Workers per job (default: 8)"
    echo "  --time HH:MM:SS           Job time limit (default: 6:00:00)"
    echo "  --mem-per-worker AMOUNT   RAM per worker, e.g. 2GB (default: 2GB)"
    echo "  --dry-run                 Print sbatch commands without submitting"
    exit 1
fi

SOURCE="$1"
FILTERED="$2"
IRREGULAR="$3"
shift 3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --workers-per-job)
            WORKERS_PER_JOB="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem-per-worker)
            MEM_PER_WORKER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate paths
for path in "$SOURCE" "$(dirname "$FILTERED")" "$(dirname "$IRREGULAR")"; do
    if [[ ! -d "$path" ]]; then
        echo "Error: Directory does not exist: $path"
        exit 1
    fi
done

echo "================================================================"
echo "Preprocessing Job Submission Configuration"
echo "================================================================"
echo "Source:           $SOURCE"
echo "Output Filtered:  $FILTERED"
echo "Output Irregular: $IRREGULAR"
echo "Number of jobs:   $NUM_JOBS"
echo "Workers per job:  $WORKERS_PER_JOB"
echo "RAM per worker:   $MEM_PER_WORKER"
echo "Time limit:       $TIME_LIMIT"
echo "Dry run:          $DRY_RUN"
echo "================================================================"
echo ""

# Submit jobs
JOB_IDS=()
for job_idx in $(seq 0 $((NUM_JOBS - 1))); do
    JOB_NAME="preprocess_${job_idx}"
    
    # Calculate total memory for this job
    TOTAL_MEM=$(awk -v workers=$WORKERS_PER_JOB -v mem_per_worker=$MEM_PER_WORKER \
        'BEGIN { 
            gsub(/[^0-9]/, "", mem_per_worker)
            print workers * mem_per_worker "G"
        }')
    
    CMD="sbatch \
        --job-name=$JOB_NAME \
        --cpus-per-task=$WORKERS_PER_JOB \
        --mem=$TOTAL_MEM \
        --time=$TIME_LIMIT \
        --output=${SCRIPT_DIR}/../logs/${JOB_NAME}_%j.log \
        --error=${SCRIPT_DIR}/../logs/${JOB_NAME}_%j.err \
        --wrap=\"python ${SCRIPT_DIR}/preprocess_images.py \
            --source '$SOURCE' \
            --output-filtered '$FILTERED' \
            --output-irregular '$IRREGULAR' \
            --workers $WORKERS_PER_JOB \
            --shard $job_idx $NUM_JOBS\""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] $CMD"
    else
        echo "Submitting job $((job_idx + 1))/$NUM_JOBS..."
        JOB_ID=$(eval "$CMD" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+")
        JOB_IDS+=("$JOB_ID")
        echo "  Job ID: $JOB_ID"
    fi
done

if [[ "$DRY_RUN" != "true" ]]; then
    echo ""
    echo "================================================================"
    echo "Jobs submitted: ${#JOB_IDS[@]}"
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor with: squeue --job ${JOB_IDS[0]}"
    echo "Cancel all:   scancel ${JOB_IDS[*]}"
    echo "================================================================"
fi
