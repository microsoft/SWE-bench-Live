# this script run all steps in the curation stage.

mkdir -p output

# Iterate through Python, C, C++, C#, Go, Java, JavaScript, TypeScript, Rust
for language in Python C C++ Go Java Rust "C#" JavaScript TypeScript; do
    echo "Processing language: $language"

    mkdir -p "output/$language"

    if [ -f "/home/v-kenanli/SWE-bench-Live/curation/output/$language/raw_repos.jsonl" ]; then echo 'skip repo crawl'; else python crawl_repo.py --language "'$language'" --min_stars 2000 --max_stars 100000 --tokens_file tokens.txt --output_file "output/$language/raw_repos.jsonl"; fi;

    if [ -f "/home/v-kenanli/SWE-bench-Live/curation/output/$language/filtered_repos.jsonl" ]; then echo 'skip repo filter'; else python filter_repo.py --input_file "output/$language/raw_repos.jsonl" --output_file "output/$language/filtered_repos.jsonl" --tokens_file tokens.txt --language "'$language'" --max_workers 10; fi;

    rm -rf job_status

    mkdir -p job_status

    if [ -f "/home/v-kenanli/SWE-bench-Live/curation/output/$language/raw_tasks.jsonl" ]; then echo 'skip pr crawl'; else bash ./swe_task_crawling/run_get_tasks_pipeline.sh --repos-jsonl "output/$language/filtered_repos.jsonl" --token-file tokens.txt --cutoff-date 20260320 --path-prs "output/$language/prs" --path-tasks "output/$language/tasks" --output-dir "output/$language/split_jobs"; fi;

    if [ -f "/home/v-kenanli/SWE-bench-Live/curation/output/$language/raw_tasks.jsonl" ]; then echo 'skip pr merge'; else python swe_task_crawling/merge_tasks.py --input_folder "output/$language/tasks" --input_repos "output/$language/filtered_repos.jsonl" --output "output/$language/raw_tasks.jsonl"; fi;

    python -m llm_filter.verify --input_dir "output/$language/raw_tasks.jsonl"  --output_dir "output/$language/verified_tasks.jsonl"  --model "azure/gpt-5.2-20251211"

    python -m llm_filter.split_os --input_file "output/$language/verified_tasks.jsonl" --windows_file "output/$language/windows_tasks.jsonl" --general_file "output/$language/general_tasks.jsonl" --model "azure/gpt-5.2-20251211"

    echo "Completed processing for $language"
    echo "-----------------------------------"
done
~                                                                                                                                                              
~          