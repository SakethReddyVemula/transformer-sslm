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

# Set PyTorch memory config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_debug.log")
    ]
)
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, HfApi
# Import MorphScore
# Assuming this script is in transformer-sslm/morphscore/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from morphscore import MorphScore

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
    # transformer-sslm/fairseq/models
    # this script: transformer-sslm/morphscore/eva..
    # custom models: ../fairseq/models
    
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

def get_model_segmentation(model, task, text, device):
    """
    Segmentation logic adapated from app.py
    """
    # Simple regex for char tokenization if task doesn't support it directly in a way we want
    # But we follow app.py logic
    tgt_dict = task.target_dictionary
    
    def char_tokenize(line):
        line = line.strip()
        return list(line)
        
    # 1. Tokenize
    # app.py puts EOS at end
    tokens = tgt_dict.encode_line(
        text, line_tokenizer=char_tokenize, add_if_not_exist=False, append_eos=False
    ).long()
    
    eos_token = tgt_dict.eos()
    target_tokens = torch.cat([tokens, torch.tensor([eos_token])])
    target_tokens = target_tokens.unsqueeze(0).to(device)
    
    # 2. Construct Input (SOS + Chars)
    sos_token = tgt_dict.eos()
    sos_tensor = torch.full((1, 1), sos_token, dtype=torch.long, device=device)
    prev_output_tokens = torch.cat([sos_tensor, target_tokens], dim=1)
    
    # 3. Model Forward
    try:
        with torch.no_grad():
            split_indices = model(prev_output_tokens, mode="segment")
    except TypeError:
        logging.warning(f"Model does not support mode='segment' for input: {text}")
        return [text] # Fallback
        
    # 4. Decode
    # We use manual reconstruction logic from app.py for stability
    raw_text = ""
    target_list = target_tokens[0].cpu().tolist()
    if target_list[-1] == eos_token:
        valid_indices = target_list[:-1]
    else:
        valid_indices = target_list
        
    for idx in valid_indices:
        raw_text += tgt_dict.symbols[idx]
        
    raw_text = raw_text.replace("</s>", " ")
    
    indices = split_indices[0]
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
        
    segments = []
    last_idx = 0
    for end_idx in indices:
        seg = raw_text[last_idx : end_idx + 1]
        segments.append(seg)
        last_idx = end_idx + 1
        
    # Filter empty
    segments = [s for s in segments if s]
    return segments, indices

def get_ckpt_num(filename):
    """
    Parses checkpoint filename to return a tuple (epoch, updates) for sorting.
    Handles:
    - checkpoint_E_U.pt (update-based) -> (E, U)
    - checkpointE.pt (epoch-based) -> (E, float('inf')) so it comes last in that epoch
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

def evaluate_checkpoint(model, task, device, dataset, morph_score_obj):
    """
    Evaluates a single model against the dataset.
    Returns: {recall, precision, f1}
    """
    
    # Storage for calculating means
    points_morphscore_recall = []
    points_morphscore_precision = []
    weights = []
    
    # Logic extracted/adapted from MorphScore.get_morphscore
    
    for idx in range(len(dataset)):
        row = dataset.iloc[idx]
        prefix = row['preceding_part']
        suffix = row['following_part']
        stem = row['stem']
        wordform = row['wordform']
        norm_freq = float(row['word_freq_norm'])
        
        if not isinstance(wordform, str):
            continue
        wordform = wordform.strip()
        if not wordform:
            continue
            
        # Assemble gold morphemes
        morphemes = []
        if not pd.isna(prefix):
            morphemes.append(str(prefix))
        morphemes.append(str(stem))
        if not isinstance(suffix, float):
            morphemes.append(str(suffix))
            
        # Predict segmentation
        try:
            pred_segments, raw_indices = get_model_segmentation(model, task, wordform, device)
            
            # Log first few examples for debugging
            if idx < 5:
                logging.info(f"Sample {idx}: '{wordform}' -> {pred_segments} (Indices: {raw_indices})")
                
        except Exception as e:
            # print(f"Error segmenting {wordform}: {e}")
            pred_segments = [wordform]
            if idx < 5:
                 logging.error(f"Error segmenting sample {idx} '{wordform}': {e}")
            
        # Score
        point_recall, point_precision = morph_score_obj.morph_eval(morphemes, pred_segments)
        
        # Weighting
        weight = 1.0
        if morph_score_obj.config['freq_scale']:
             weight = norm_freq
             
        if not np.isnan(point_recall):
            points_morphscore_recall.append(point_recall * weight)
            points_morphscore_precision.append(point_precision * weight)
            weights.append(weight)
            
    # Calculate means
    total_weight = sum(weights) if weights else 0
    if total_weight == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0}
        
    mean_recall = sum(points_morphscore_recall) / total_weight
    mean_precision = sum(points_morphscore_precision) / total_weight
    
    if (mean_precision + mean_recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
        
    return {'precision': mean_precision, 'recall': mean_recall, 'f1': f1}

def process_epoch(epoch, filename, args, dataset, ms, hf_token):
    logging.info(f"Processing Checkpoint {epoch} ({filename})...")
    print(f"Processing Checkpoint {epoch} ({filename})...")
    
    ckpt_path = None
    model = None
    task = None
    models = None
    
    try:
        # Download
        try:
            ckpt_path = hf_hub_download(repo_id=args.repo_id, filename=filename, token=hf_token, local_dir="temp_eval_ckpts")
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
        scores = evaluate_checkpoint(model, task, device, dataset, ms)
        print(f"Checkpoint {epoch}: P={scores['precision']:.4f}, R={scores['recall']:.4f}, F1={scores['f1']:.4f}")
        logging.info(f"Checkpoint {epoch} Scores: {scores}")

        return scores['precision'], scores['recall'], scores['f1']

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
    parser.add_argument("--lang", type=str, required=True, help="Language code for SSLM loading (e.g. hin)")
    parser.add_argument("--lang-long", type=str, required=True, help="Language code for data file (e.g. hindi)")
    parser.add_argument("--script", type=str, default="deva", help="Script name")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing data files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token")
    args = parser.parse_args()
    
    # Get token from arg or env
    hf_token = args.token if args.token else os.environ.get("HF_TOKEN")

    
    setup_fairseq_modules()
    
    # 1. Load Data
    data_path = os.path.join(args.data_dir, f"{args.lang_long}_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print(f"Please place the {args.lang_long}_data.csv file in {os.path.abspath(args.data_dir)}")
        sys.exit(1)
        
    print(f"Loading data from {data_path}...")
    
    # Initialize MorphScore to reuse its loading/filtering logic
    # We point data_dir to args.data_dir so it finds the file
    ms = MorphScore(data_dir=args.data_dir, language_subset=[args.lang_long])
    
    # Load and filter
    # _load_dataset expects lang name or filename inside data_dir
    # user said data would be data/{LANG_LONG}_data.csv
    try:
        dataset = ms._load_dataset(f"{args.lang_long}_data.csv")
        dataset = ms._filter_dataset(dataset)
        print(f"Loaded {len(dataset)} samples after filtering.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
        
    # 2. List Checkpoints
    print(f"Fetching checkpoints from {args.repo_id}...")
    api = HfApi(token=hf_token)
    try:
        files = api.list_repo_files(repo_id=args.repo_id)
    except Exception as e:
        print(f"Error accessing repository: {e}")
        print("Please ensure your HF_TOKEN is valid and has access to the repository.")
        sys.exit(1)

    
    # Filter for checkpoints
    # Pattern: checkpointN.pt
    ckpt_files = []
    for f in files:
        # Filter by language if provided
        if args.lang:
             # Check if file is in a directory named args.lang or starts with it
             if not (f.startswith(f"{args.lang}/") or f"/{args.lang}/" in f):
                 continue

        if f.endswith(".pt"):
             num_tuple = get_ckpt_num(f)
             if num_tuple[0] > 0: 
                 ckpt_files.append((num_tuple, f))
                 
    # Sort by epoch
    ckpt_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(ckpt_files)} checkpoints.")
    
    epochs = []
    checkpoint_names = []
    precisions = []
    recalls = []
    f1s = []
    
    # 3. Evaluate Loop
    for (epoch, updates), filename in ckpt_files:
        # Pass a friendly label for logging
        if updates == float('inf'):
             label = f"Epoch {epoch}"
        else:
             label = f"Epoch {epoch} (Step {updates})"
             
        result = process_epoch(label, filename, args, dataset, ms, hf_token)
        
        # Explicit GC after function return
        gc.collect()
        torch.cuda.empty_cache()
        
        if result:
            p, r, f1 = result
            # For plotting, use a composite x-axis value or just a string label?
            # Matplotlib handles string labels on x-axis fine if they are unique/sorted.
            # But let's use the label constructed above
            if updates == float('inf'):
                 x_val = f"E{epoch}"
            else:
                 x_val = f"E{epoch}_S{updates}"
                 
            epochs.append(x_val)
            checkpoint_names.append(filename)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)


    # Save to CSV
    results_df = pd.DataFrame({
        'checkpoint_name': checkpoint_names,
        'precision': precisions,
        'recall': recalls,
        'f1': f1s
    })
    results_csv = "morphscore_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, precisions, label='Precision', marker='o')
    plt.plot(epochs, recalls, label='Recall', marker='s')
    plt.plot(epochs, f1s, label='F1-Score', marker='^')
    
    plt.xlabel('Checkpoint Step/Epoch')
    plt.ylabel('Score')
    plt.title(f'MorphScore Dynamics - {args.lang_long}')
    plt.legend()
    plt.grid(True)
    
    out_file = "morphscore_evolution.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
