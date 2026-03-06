import argparse
import os
import sys
import re
import json
import torch
import gc
from collections import Counter
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi

# Set PyTorch memory config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup Fairseq path BEFORE importing fairseq
fairseq_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fairseq"))
if os.path.exists(fairseq_repo_path) and fairseq_repo_path not in sys.path:
    sys.path.insert(0, fairseq_repo_path)

from fairseq import checkpoint_utils

def setup_fairseq_modules():
    import fairseq
    import fairseq.models
    import importlib.util

    custom_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fairseq/models"))
    
    ssmt_path = os.path.join(custom_models_dir, "ssmt/__init__.py")
    if os.path.exists(ssmt_path):
        spec = importlib.util.spec_from_file_location("fairseq.models.ssmt", ssmt_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["fairseq.models.ssmt"] = module
        fairseq.models.ssmt = module
        spec.loader.exec_module(module)

    model_file_path = os.path.join(custom_models_dir, "transformer_sslm.py")
    if os.path.exists(model_file_path):
        if "transformer_sslm" not in fairseq.models.MODEL_REGISTRY:
            spec = importlib.util.spec_from_file_location("transformer_sslm_custom", model_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

def get_model_segmentation(model, task, text_list, device, batch_size=512):
    results = {}
    tgt_dict = task.target_dictionary
    eos_token = tgt_dict.eos()
    sos_token = tgt_dict.eos() 
    pad_token = tgt_dict.pad()

    def char_tokenize(line):
        return list(line.strip())

    for i in tqdm(range(0, len(text_list), batch_size), desc="Segmenting", leave=False):
        batch_texts = text_list[i:i+batch_size]
        batch_tokens = []
        batch_lengths = []

        for text in batch_texts:
            tokens = tgt_dict.encode_line(
                text, line_tokenizer=char_tokenize, add_if_not_exist=False, append_eos=False
            ).long()
            batch_tokens.append(tokens)
            batch_lengths.append(tokens.size(0))
        
        # Enforce min length to avoid "stack expects a non-empty TensorList" error
        max_len = max(max(batch_lengths) + 1, 10) 
        
        padded_batch = []
        for tokens in batch_tokens:
            t = torch.cat([tokens, torch.tensor([eos_token], dtype=torch.long)])
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                pad_tensor = torch.full((pad_len,), pad_token, dtype=torch.long)
                t = torch.cat([t, pad_tensor])
            padded_batch.append(t)
            
        target_tokens = torch.stack(padded_batch).to(device)
        sos_tensor = torch.full((target_tokens.size(0), 1), sos_token, dtype=torch.long, device=device)
        prev_output_tokens = torch.cat([sos_tensor, target_tokens], dim=1)
        
        try:
            with torch.no_grad():
                batch_split_indices = model(prev_output_tokens, mode="segment")
        except Exception as e:
            print(f"Model batch failed, error: {e}")
            for text in batch_texts:
                results[text] = [text]
            continue

        for j, text in enumerate(batch_texts):
            raw_text = ""
            for t_id in target_tokens[j].cpu().tolist():
                if t_id == eos_token: break
                raw_text += tgt_dict.symbols[t_id]
            raw_text = raw_text.replace("</s>", " ")
            
            indices = batch_split_indices[j]
            if isinstance(indices, torch.Tensor): 
                indices = indices.tolist()
            
            segments = []
            last_idx = 0
            for end_idx in indices:
                if end_idx >= len(raw_text): break
                seg = raw_text[last_idx : end_idx + 1]
                segments.append(seg)
                last_idx = end_idx + 1
            
            segments = [s for s in segments if s]
            if last_idx < len(raw_text):
                 segments.append(raw_text[last_idx:])
            
            results[text] = segments

    return results

def get_checkpoints(args):
    api = HfApi(token=args.token)
    try:
        files = api.list_repo_files(repo_id=args.repo_id)
    except Exception as e:
        print(f"Error accessing repository {args.repo_id}: {e}")
        return []
        
    candidates = [f for f in files if f.endswith(".pt")]
    if args.lang:
         candidates = [f for f in candidates if f.startswith(f"{args.lang}/") or f"/{args.lang}/" in f]

    def sort_key(f):
        base = os.path.basename(f)
        # m = re.search(r'checkpoint_(\d+)_(\d+)\.pt$', base)
        # if m: return (int(m.group(1)), int(m.group(2)))
        # m = re.search(r'checkpoint(\d+)\.pt$', base)
        m = re.search(r'checkpoint(2[2-5])\.pt$', base)
        if m: return (int(m.group(1)), 999999)
        return (-1, -1)
        
    candidates.sort(key=sort_key)
    # filter valid checkpoints
    return [(os.path.basename(f), f) for f in candidates if sort_key(f)[0] >= 0] 

pending_uploads = []

def queue_for_upload(file_path):
    pending_uploads.append(file_path)

def flush_uploads(repo_id, token, subfolder=None, delete_local=False, force=False, batch_size=3):
    if not pending_uploads: return
    if not force and len(pending_uploads) < batch_size: return

    try:
        from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError:
        print("huggingface_hub not installed.")
        return

    api = HfApi(token=token)
    operations = []
    files_to_upload = list(pending_uploads)

    for fp in files_to_upload:
        path_in_repo = os.path.basename(fp)
        if subfolder:
            path_in_repo = f"{subfolder}/{path_in_repo}"
        operations.append(CommitOperationAdd(path_or_fileobj=fp, path_in_repo=path_in_repo))

    try:
        filenames = [os.path.basename(fp) for fp in files_to_upload]
        print(f"Batch uploading {len(files_to_upload)} files to {repo_id}...")
        api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=f"Upload {len(files_to_upload)} segmentations: {', '.join(filenames)}",
            repo_type="dataset"
        )
        print(f"Successfully uploaded: {', '.join(filenames)}")
        for fp in files_to_upload:
            pending_uploads.remove(fp)
            if delete_local and os.path.exists(fp):
                os.remove(fp)
    except Exception as e:
        print(f"Failed batch upload: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="HF Repo ID for Checkpoints")
    parser.add_argument("--upload-repo-id", type=str, default=None, help="HF Repo ID to upload results to")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g. hin)")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--data-path", type=str, default=None, help="Explicit path to testing/validation file.")
    parser.add_argument("--output-dir", type=str, default="segmentations_eval")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--keep-downloaded", action="store_true", help="Keep downloaded checkpoints")
    args = parser.parse_args()
    
    setup_fairseq_modules()
    
    # 1. Load Data
    if args.data_path:
        data_file = args.data_path
    else:
        # Construct default path checking test first, then valid
        home = os.path.expanduser("~")
        data_file = os.path.join(home, "dataset", args.lang, f"test.{args.lang}")
        if not os.path.exists(data_file):
            data_file = os.path.join(home, "dataset", args.lang, f"valid.{args.lang}")
            
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)
            
    print(f"Loading data from {data_file}...")
    word_counts = Counter()
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                word_counts.update(words)
    except Exception as e:
        print(f"Error reading data file: {e}")
        sys.exit(1)
            
    unique_words = list(word_counts.keys())
    print(f"Loaded {len(unique_words)} unique words from dataset.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
    
    # 2. List Checkpoints
    print(f"Fetching checkpoints from {args.repo_id}...")
    checkpoints = get_checkpoints(args)
    print(f"Found {len(checkpoints)} checkpoints.")
    
    for ckpt_name, ckpt_ref in tqdm(checkpoints, desc="Checkpoints"):
        output_filename = f"seg_{args.lang}_{ckpt_name}.json"
        output_file = os.path.join(args.output_dir, output_filename)
        
        if os.path.exists(output_file):
            print(f"Skipping {ckpt_name}, already exists at {output_file}")
            continue
            
        print(f"Processing {ckpt_name}...")
        
        # Download Checkpoint
        local_ckpt_dir = os.path.join(args.output_dir, "temp_checkpoints")
        os.makedirs(local_ckpt_dir, exist_ok=True)
        try:
            ckpt_path = hf_hub_download(
                repo_id=args.repo_id, filename=ckpt_ref, token=args.token,
                local_dir=local_ckpt_dir, local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"Error downloading {ckpt_ref}: {e}")
            continue

        model, task, models = None, None, None
        try:
            models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
            model = models[0]
            model.eval()
            if torch.cuda.is_available(): model.cuda()
            device = next(model.parameters()).device
            
            segmentations = get_model_segmentation(model, task, unique_words, device, batch_size=args.batch_size)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(segmentations, f, ensure_ascii=False, indent=2)
                
            print(f"Saved to {output_file}")
            manifest.append({"checkpoint": ckpt_name, "file": output_filename})
            with open(manifest_path, 'w') as f: json.dump(manifest, f, indent=2)
                
            if args.upload_repo_id:
                queue_for_upload(output_file)
                flush_uploads(args.upload_repo_id, args.token, args.lang, delete_local=True, batch_size=3)

        except Exception as e:
            print(f"Failed to process {ckpt_name}: {e}")
        finally:
            if model: del model
            if task: del task
            if models: del models
            torch.cuda.empty_cache()
            gc.collect()
            
            if not args.keep_downloaded and os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    if args.upload_repo_id:
        flush_uploads(args.upload_repo_id, args.token, args.lang, delete_local=True, force=True)
        
if __name__ == "__main__":
    main()
