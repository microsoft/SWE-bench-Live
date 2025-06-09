#!/bin/bash

# Optimized one-call SWE task crawling pipeline
# Automatically splits repos based on available tokens and runs the pipeline

# Default values
REPOS_JSONL=""
TOKEN_FILE=""
OUTPUT_DIR="splitted_jobs"
PATH_PRS="../prs"
PATH_TASKS="../tasks"
CUTOFF_DATE="20250101"
MAX_PULLS=""
DRY_RUN=false

# Function to display usage
usage() {
    echo "Usage: $0 --repos-jsonl <path> --token-file <path> [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --repos-jsonl    Path to JSONL file containing repository list"
    echo "  --token-file     Path to file containing GitHub tokens (one per line)"
    echo ""
    echo "Optional arguments:"
    echo "  --output-dir     Directory to store split job files (default: splitted_jobs)"
    echo "  --path-prs       Path to save PR data files (default: ../prs)"
    echo "  --path-tasks     Path to save task instance data files (default: ../tasks)"
    echo "  --cutoff-date    Cutoff date for PRs in YYYYMMDD format (default: 20250101)"
    echo "  --max-pulls      Maximum number of pulls to process (optional)"
    echo "  --dry-run        Only split repos and show what would be executed"
    echo "  --help           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --repos-jsonl repos.jsonl --token-file tokens.txt --cutoff-date 20240101"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repos-jsonl)
            REPOS_JSONL="$2"
            shift 2
            ;;
        --token-file)
            TOKEN_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --path-prs)
            PATH_PRS="$2"
            shift 2
            ;;
        --path-tasks)
            PATH_TASKS="$2"
            shift 2
            ;;
        --cutoff-date)
            CUTOFF_DATE="$2"
            shift 2
            ;;
        --max-pulls)
            MAX_PULLS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$REPOS_JSONL" || -z "$TOKEN_FILE" ]]; then
    echo "Error: Both --repos-jsonl and --token-file are required"
    usage
fi

# Validate input files exist
if [[ ! -f "$REPOS_JSONL" ]]; then
    echo "Error: Repository JSONL file not found: $REPOS_JSONL"
    exit 1
fi

if [[ ! -f "$TOKEN_FILE" ]]; then
    echo "Error: Token file not found: $TOKEN_FILE"
    exit 1
fi

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/job_status"
mkdir -p "$PATH_PRS"
mkdir -p "$PATH_TASKS"

echo "🚀 Starting SWE task crawling pipeline"
echo "📁 Repository list: $REPOS_JSONL"
echo "🔑 Token file: $TOKEN_FILE"
echo "📂 Split jobs directory: $OUTPUT_DIR"
echo "💾 PRs will be saved to: $PATH_PRS"
echo "📋 Tasks will be saved to: $PATH_TASKS"
echo "📅 Cutoff date: $CUTOFF_DATE"

# Step 1: Split repositories based on available tokens
echo ""
echo "🔀 Step 1: Splitting repositories..."

python "$(dirname "$0")/split_jobs.py" "$REPOS_JSONL" "$TOKEN_FILE" "$OUTPUT_DIR"

if [[ $? -ne 0 ]]; then
    echo "❌ Error: Failed to split repositories"
    exit 1
fi

# Count tokens to determine which token IDs to use
TOKEN_COUNT=$(wc -l < "$TOKEN_FILE" | tr -d ' ')
TOKEN_IDS=($(seq 0 $((TOKEN_COUNT-1))))

echo "🎯 Will use token IDs: ${TOKEN_IDS[*]}"

# If dry run, show what would be executed and exit
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "🔍 DRY RUN - Would execute the following commands:"
    for token_id in "${TOKEN_IDS[@]}"; do
        echo "python get_tasks_pipeline.py \\"
        echo "    --gh_token_file $TOKEN_FILE \\"
        echo "    --token_ids $token_id \\"
        echo "    --repo_list_file $OUTPUT_DIR/token-${token_id}.txt \\"
        echo "    --path_prs $PATH_PRS \\"
        echo "    --path_tasks $PATH_TASKS \\"
        echo "    --cutoff_date $CUTOFF_DATE \\"
        if [[ -n "$MAX_PULLS" ]]; then
            echo "    --max_pulls $MAX_PULLS \\"
        fi
        echo "    &"
        echo ""
    done
    echo "# Wait for all jobs to complete"
    echo "wait"
    exit 0
fi

# Step 2: Run the pipeline for each token
echo ""
echo "🎯 Step 2: Starting parallel processing..."

# Function to handle script termination
terminate() {
    echo "⚠️  Termination signal received, cleaning up..."
    rm -rf "$OUTPUT_DIR/job_status"
    pkill -P $$
    pkill -f get_tasks_pipeline.py
    exit 0
}

trap 'terminate' SIGINT SIGTERM

# Function to start crawling for a specific token
start_crawling() {
    local token_id=$1

    echo "🚀 Starting processing with token ID $token_id"

    local cmd="python $(dirname "$0")/get_tasks_pipeline.py"
    cmd="$cmd --gh_token_file $TOKEN_FILE"
    cmd="$cmd --token_ids $token_id"
    cmd="$cmd --repo_list_file $OUTPUT_DIR/token-${token_id}.txt"
    cmd="$cmd --path_prs $PATH_PRS"
    cmd="$cmd --path_tasks $PATH_TASKS"
    cmd="$cmd --cutoff_date $CUTOFF_DATE"
    
    if [[ -n "$MAX_PULLS" ]]; then
        cmd="$cmd --max_pulls $MAX_PULLS"
    fi

    echo "📝 Executing: $cmd"
    
    eval "$cmd" &
    local pid=$!
    echo "🏃 Started crawling with PID $pid for token ID $token_id"
    echo $pid > "$OUTPUT_DIR/job_status/pid_${token_id}.txt"

    wait $pid
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Successfully completed processing for token ID $token_id"
    else
        echo "❌ Error encountered with token ID $token_id (exit code: $exit_code)"
        return $exit_code
    fi
}

# Launch crawling for all tokens in parallel
launch_all_crawling() {
    local pids=()
    
    for token_id in "${TOKEN_IDS[@]}"; do
        start_crawling $token_id &
        pids+=($!)
    done
    
    echo "⏳ Waiting for all ${#pids[@]} jobs to complete..."
    
    # Wait for all background jobs and collect exit codes
    local failed_jobs=0
    for pid in "${pids[@]}"; do
        wait $pid
        if [[ $? -ne 0 ]]; then
            ((failed_jobs++))
        fi
    done
    
    if [[ $failed_jobs -eq 0 ]]; then
        echo "🎉 All jobs completed successfully!"
    else
        echo "⚠️  $failed_jobs job(s) failed"
        return 1
    fi
}

# Execute the pipeline
launch_all_crawling

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🏆 Pipeline completed successfully!"
    echo "📁 PR data saved to: $PATH_PRS"
    echo "📋 Task data saved to: $PATH_TASKS"
else
    echo ""
    echo "💥 Pipeline completed with errors"
    exit 1
fi 