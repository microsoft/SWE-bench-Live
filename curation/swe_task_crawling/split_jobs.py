import os
import json
import random
import argparse

def split_repos(repos_jsonl, token_file, output_dir):
    with open(repos_jsonl, 'r') as f:
        repos = [json.loads(line) for line in f]
        repos = [repo['full_name'] for repo in repos]

    with open(token_file, 'r') as f:
        tokens = [line.strip() for line in f if line.strip()]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    token_count = len(tokens)
    chunk_size = len(repos) // token_count

    print(f"📊 Found {len(repos)} repositories and {token_count} tokens")
    print(f"📦 Splitting into {token_count} chunks of ~{chunk_size} repos each")

    for token_id in range(token_count):
        start_index = token_id * chunk_size
        end_index = start_index + chunk_size if token_id != token_count - 1 else len(repos)
        token_repos = repos[start_index:end_index]
        filename = f"token-{token_id}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(token_repos))
        
        print(f"✅ Created {filepath} with {len(token_repos)} repositories")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split repositories into chunks based on available tokens")
    parser.add_argument("repos_jsonl", help="Path to JSONL file containing repository list")
    parser.add_argument("token_file", help="Path to file containing GitHub tokens (one per line)")
    parser.add_argument("output_dir", help="Directory to store split job files")
    
    args = parser.parse_args()
    split_repos(args.repos_jsonl, args.token_file, args.output_dir)
