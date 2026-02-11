import argparse
import os
import sys
import re
import glob
import json
import torch
import gc
import shutil
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, HfApi
from collections import Counter
from tqdm import tqdm

# Set PyTorch memory config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fertility_evaluation.log")
    ]
)

# Setup Fairseq path BEFORE importing fairseq
fairseq_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fairseq"))
if os.path.exists(fairseq_repo_path):
    if fairseq_repo_path not in sys.path:
        sys.path.insert(0, fairseq_repo_path)

from fairseq import checkpoint_utils

# Setup Fairseq modules (custom models)
def setup_fairseq_modules():
    import fairseq
    import fairseq.models
    import importlib.util

    # 2. Inject 'ssmt' and 'transformer_sslm' into fairseq.models
    custom_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fairseq/models"))
    
    # Inject ssmt
    ssmt_path = os.path.join(custom_models_dir, "ssmt/__init__.py")
    if os.path.exists(ssmt_path):
        spec = importlib.util.spec_from_file_location("fairseq.models.ssmt", ssmt_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["fairseq.models.ssmt"] = module
        fairseq.models.ssmt = module
        spec.loader.exec_module(module)

    # Load transformer_sslm (registers the model)
    model_file_path = os.path.join(custom_models_dir, "transformer_sslm.py")
    if os.path.exists(model_file_path):
        if "transformer_sslm" not in fairseq.models.MODEL_REGISTRY:
            spec = importlib.util.spec_from_file_location("transformer_sslm_custom", model_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

def get_model_segmentation(model, task, text_list, device, batch_size=512):
    """
    Batched segmentation logic.
    """
    results = {}
    tgt_dict = task.target_dictionary
    eos_token = tgt_dict.eos()
    sos_token = tgt_dict.eos() 
    pad_token = tgt_dict.pad()

    def char_tokenize(line):
        line = line.strip()
        return list(line)

    for i in tqdm(range(0, len(text_list), batch_size), desc="Segmenting", leave=False):
        batch_texts = text_list[i:i+batch_size]
        batch_tokens = []
        batch_lengths = []

        # 1. Tokenize and measure
        for text in batch_texts:
            tokens = tgt_dict.encode_line(
                text, line_tokenizer=char_tokenize, add_if_not_exist=False, append_eos=False
            ).long()
            batch_tokens.append(tokens)
            batch_lengths.append(tokens.size(0))
        
        # 2. Pad
        # We need to append EOS to each, then pad.
        # Max length including EOS
        max_len = max(batch_lengths) + 1 
        # Enforce min length to avoid "stack expects a non-empty TensorList" error in model
        max_len = max(max_len, 10) 
        
        padded_batch = []
        for tokens in batch_tokens:
            # Add EOS
            t = torch.cat([tokens, torch.tensor([eos_token], dtype=torch.long)])
            curr_len = t.size(0)
            pad_len = max_len - curr_len
            if pad_len > 0:
                pad_tensor = torch.full((pad_len,), pad_token, dtype=torch.long)
                t = torch.cat([t, pad_tensor])
            padded_batch.append(t)
            
        target_tokens = torch.stack(padded_batch).to(device) # (batch, max_len)
        
        # 3. Construct Input (SOS + tokens)
        bs = target_tokens.size(0)
        sos_tensor = torch.full((bs, 1), sos_token, dtype=torch.long, device=device)
        prev_output_tokens = torch.cat([sos_tensor, target_tokens], dim=1) # (batch, max_len+1)
        
        # 4. Model Forward (Batched)
        try:
            with torch.no_grad():
                # transformer_sslm support batch processing in segment mode
                # It returns a list of lists (split_indices)
                batch_split_indices = model(prev_output_tokens, mode="segment")
        except Exception as e:
            logging.warning(f"Model batch failed, error: {e}")
            # Fallback: one by one or skip
            for text in batch_texts:
                results[text] = [text]
            continue

        # 5. Decode
        for j, text in enumerate(batch_texts):
            # Reconstruct raw text from valid tokens (up to EOS)
            # We can use the original text or reconstruction. Reconstruction handles unk properly if any.
            # But char_tokenize + encode_line usually preserves chars unless unk.
            # Let's use reconstruction to be safe with model/vocab.
            
            # This is reconstruction from tokenizer indices
            # ignoring EOS and PAD
            raw_text = ""
            target_row = target_tokens[j].cpu().tolist()
            
            valid_indices = []
            for t_id in target_row:
                if t_id == eos_token:
                    break
                valid_indices.append(t_id)
            
            for idx in valid_indices:
                raw_text += tgt_dict.symbols[idx]
            
            raw_text = raw_text.replace("</s>", " ")
            
            indices = batch_split_indices[j]
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            
            segments = []
            last_idx = 0
            for end_idx in indices:
                if end_idx >= len(raw_text):
                    break
                seg = raw_text[last_idx : end_idx + 1]
                segments.append(seg)
                last_idx = end_idx + 1
            
            # Clean up empty segments
            segments = [s for s in segments if s]
            
            # Verify? If segments join to raw_text
            # Sometimes last segment might be missing if end_idx < len-1? 
            # App.py logic usually assumes split_indices covers it.
            # If not, append remainder?
            if last_idx < len(raw_text):
                 # This implies the last segment didn't end at the very end of string.
                 # Usually Viterbi ensures path to end.
                 # But if we broke due to len check, we might have remainder.
                 # Or if Viterbi logic for "end" differs.
                 # Let's append remainder to be safe.
                 segments.append(raw_text[last_idx:])
            
            results[text] = segments

    return results

def get_ckpt_num(filename):
    """
    Parses checkpoint filename to return a tuple (epoch, updates) for sorting.
    """
    # Try update format: checkpoint_1_500.pt
    match_update = re.search(r'checkpoint_(\d+)_(\d+)\.pt', filename)
    if match_update:
        return (int(match_update.group(1)), int(match_update.group(2)))
        
    # Try epoch format: checkpoint1.pt
    match_epoch = re.search(r'checkpoint(\d+)\.pt', filename)
    if match_epoch:
        return (int(match_epoch.group(1)), float('inf'))
        
    return (-1, -1)

def evaluate_fertility(model, task, device, word_counts):
    """
    Evaluates fertility on the dataset.
    Fertility = Average number of subwords per word.
    Weighted by word frequency.
    """
    total_subwords = 0
    total_words = 0
    
    unique_words = list(word_counts.keys())
    
    # Batch processing could be added here if needed, but for now simple loop
    # We can process in chunks to show progress if needed
    
    # Get segmentations
    # To improve speed, we can batch? 
    # For now, let's just loop or use the batched function if we implemented it fully batched.
    # The get_model_segmentation above iterates one by one inside.
    
    segmentations = get_model_segmentation(model, task, unique_words, device)
    
    for word, count in word_counts.items():
        segs = segmentations.get(word, [word])
        num_segs = len(segs)
        
        total_subwords += num_segs * count
        total_words += count
        
    if total_words == 0:
        return 0.0
        
    return total_subwords / total_words

def process_epoch(label, filename, args, word_counts, hf_token):
    logging.info(f"Processing {label} ({filename})...")
    # print(f"Processing {label} ({filename})...") # tqdm handles this better
    
    ckpt_path = None
    model = None
    task = None
    models = None
    
    try:
        # Download
        try:
            ckpt_path = hf_hub_download(repo_id=args.repo_id, filename=filename, token=hf_token, local_dir="temp_fertility_ckpts")
        except Exception as e:
            logging.error(f"Error downloading {filename}: {e}")
            return None

        # Load Model
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        model = models[0]
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        device = next(model.parameters()).device
        
        # Evaluate
        fertility = evaluate_fertility(model, task, device, word_counts)
        # print(f"{label}: Fertility = {fertility:.4f}")
        logging.info(f"{label} Fertility: {fertility}")

        return fertility

    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None
        
    finally:
        # cleanup
        if model is not None: 
            model.cpu()
            del model
        if task is not None: del task
        if models is not None: del models
        
        torch.cuda.empty_cache()
        gc.collect()
        
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                os.remove(ckpt_path)
            except OSError as e:
                logging.warning(f"Error removing checkpoint file: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Repository ID")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g. hin)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token")
    parser.add_argument("--data-path", type=str, default=None, help="Explicit path to validation file. If not set, uses ~/dataset/{lang}/valid.{lang}")
    parser.add_argument("--save_csv", type=str, default=None, help="Path to save the CSV file")
    parser.add_argument("--save_png", type=str, default=None, help="Path to save the PNG file")
    args = parser.parse_args()
    
    hf_token = args.token if args.token else os.environ.get("HF_TOKEN")
    
    setup_fairseq_modules()
    
    # 1. Load Data
    if args.data_path:
        data_file = args.data_path
    else:
        # Construct default path: ~/dataset/{lang}/valid.{lang}
        home = os.path.expanduser("~")
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
        
    print(f"Loaded {len(word_counts)} unique words from dataset.")
    
    # 2. List Checkpoints
    print(f"Fetching checkpoints from {args.repo_id}...")
    api = HfApi(token=hf_token)
    try:
        files = api.list_repo_files(repo_id=args.repo_id)
    except Exception as e:
        print(f"Error accessing repository: {e}")
        sys.exit(1)
        
    ckpt_files = []
    for f in files:
        if args.lang:
             # Check if file is in a directory named args.lang or starts with it
             # The user structure seems to be usually flat or repo per lang?
             # But evaluate_checkpoints.py had logic: f.startswith(f"{args.lang}/") or f"/{args.lang}/" in f
             # The user prompt for evaluate_checkpoints.py had --lang hin.
             # Let's assume the repo structure matches what's expected or reuse that logic.
             if not (f.startswith(f"{args.lang}/") or f"/{args.lang}/" in f):
                 continue

        if f.endswith(".pt"):
             num_tuple = get_ckpt_num(f)
             if num_tuple[0] > 0: 
                 ckpt_files.append((num_tuple, f))
                 
    ckpt_files.sort(key=lambda x: x[0])
    print(f"Found {len(ckpt_files)} checkpoints.")
    
    epochs = []
    checkpoint_names = []
    fertilities = []
    
    # 3. Evaluate Loop
    # Use tqdm here for progress per checkpoint
    for (epoch, updates), filename in tqdm(ckpt_files, desc="Checkpoints"):
        if updates == float('inf'):
             label = f"Epoch {epoch}"
             x_val = f"E{epoch}"
        else:
             label = f"Epoch {epoch} (Step {updates})"
             x_val = f"E{epoch}_S{updates}"
             
        # Tqdm handles progress display, so we can mute print inside or use tqdm.write
        # process_epoch has a print, we commented it out or it will interfere.
        # But let's leave logging.
        
        fertility = process_epoch(label, filename, args, word_counts, hf_token)
        
        # Explicit GC
        gc.collect()
        torch.cuda.empty_cache()
        
        if fertility is not None:
            epochs.append(x_val)
            checkpoint_names.append(filename)
            fertilities.append(fertility)
            
    # Save Results
    results_df = pd.DataFrame({
        'checkpoint_name': checkpoint_names,
        'fertility': fertilities
    })
    results_csv = args.save_csv
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, fertilities, label='Fertility', marker='o', color='purple')
    
    plt.xlabel('Checkpoint Step/Epoch')
    plt.ylabel('Average Fertility (subwords per word)')
    plt.title(f'Fertility Evolution - {args.lang}')
    plt.grid(True)
    plt.legend()
    
    out_file = args.save_png
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
