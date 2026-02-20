import argparse
import os
import re
import shutil
from huggingface_hub import HfApi, hf_hub_download

def get_ckpt_num(filename):
    """
    Parses checkpoint filename to return a tuple (epoch, updates) for sorting.
    """
    match_update = re.search(r'checkpoint_(\d+)_(\d+)\.pt', filename)
    if match_update:
        return (int(match_update.group(1)), int(match_update.group(2)))
        
    match_epoch = re.search(r'checkpoint(\d+)\.pt', filename)
    if match_epoch:
        return (int(match_epoch.group(1)), float('inf'))
        
    return (-1, -1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Repository ID")
    parser.add_argument("--lang", type=str, required=True, help="Language subfolder")
    parser.add_argument("--save-dir", type=str, required=True, help="Local directory to save checkpoint_last.pt")
    args = parser.parse_args()

    if not args.repo_id:
        print("No --repo-id provided. Skipping Hugging Face download.")
        return

    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)

    try:
        files = api.list_repo_files(repo_id=args.repo_id)
    except Exception as e:
        print(f"Error accessing repository {args.repo_id}: {e}")
        print("Please ensure your HF_TOKEN is valid and has access.")
        return

    ckpt_files = []
    for f in files:
        if args.lang:
            if not (f.startswith(f"{args.lang}/") or f"/{args.lang}/" in f):
                continue

        if f.endswith(".pt"):
             num_tuple = get_ckpt_num(f)
             if num_tuple[0] > 0: 
                 ckpt_files.append((num_tuple, f))
                 
    if not ckpt_files:
        print(f"No valid checkpoints found in repository for language '{args.lang}'.")
        return

    # Sort by epoch, then update step
    ckpt_files.sort(key=lambda x: x[0])
    latest_tuple, latest_filename = ckpt_files[-1]

    print(f"Latest checkpoint found: {latest_filename} (Epoch {latest_tuple[0]}, Step {latest_tuple[1]})")

    os.makedirs(args.save_dir, exist_ok=True)
    target_path = os.path.join(args.save_dir, "checkpoint_last.pt")

    print(f"Downloading to {target_path} ...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=latest_filename,
            token=hf_token
        )
        shutil.copy2(downloaded_path, target_path)
        print("Successfully downloaded and copied checkpoint_last.pt!")
    except Exception as e:
        print(f"Error downloading {latest_filename}: {e}")

if __name__ == "__main__":
    main()
