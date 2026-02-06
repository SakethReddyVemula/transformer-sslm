import streamlit as st
import torch
import os
import sys
import numpy as np
import altair as alt
import pandas as pd
from fairseq import checkpoint_utils, utils, tasks, options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import importlib.util

# Ensure fairseq is importable
# We need the 'real' fairseq (with metrics, etc.) which seems to be in ../../fairseq (sibling to transformer-sslm)
# But we also need the custom models in transformer-sslm/fairseq/models

import importlib.util

# 1. Add valid fairseq to path
fairseq_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fairseq"))
if os.path.exists(fairseq_repo_path):
    sys.path.insert(0, fairseq_repo_path)
else:
    # Fallback to current environment, maybe it's installed
    pass

import fairseq
import fairseq.models

# 2. Inject 'ssmt' and 'transformer_sslm' into fairseq.models
# Path to custom models
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
    # check if already registered to avoid duplicates on re-run
    if "transformer_sslm" not in fairseq.models.MODEL_REGISTRY:
        spec = importlib.util.spec_from_file_location("transformer_sslm_custom", model_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


# --- Helper Functions ---

@st.cache_resource
def load_model(checkpoint_path, data_path, user_dir=None):
    """Loads the SSLM model from a checkpoint."""
    try:
        # Load the model and task
        # We need to setup the task similarly to how train.py does it
        # However, checkpoint_utils usually handles this if we provide the right args
        
        # Determine the directory containing the checkpoint to find dictionaries if implied
        ckpt_dir = os.path.dirname(checkpoint_path)
        if not data_path:
             # Try to infer data path from checkpoint directory or parent
             # Often in fairseq, dict.txt is in the data-bin directory provided during training
             pass

        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path],
            arg_overrides={"data": data_path} if data_path else {}
        )
        model = models[0]
        model.eval()
        model.cuda() if torch.cuda.is_available() else model.cpu()
        return model, task, cfg
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None, None, None

def get_model_segmentation(model, task, text, device):
    """
    Uses the model's internal Viterbi logic to segment the text.
    Replicates logic from SubwordSegmentalLanguageModelingTask.valid_step
    """
    import re
    SPACE_NORMALIZER = re.compile(r"\s+")
    
    def char_tokenize(line):
        line = SPACE_NORMALIZER.sub(" ", line)
        line = line.strip()
        return list(line)

    # 1. Tokenize (Input Text -> Chars)
    tgt_dict = task.target_dictionary
    
    # Use Fairseq dictionary encode_line to map chars to indices
    # We use the same char_tokenize helper as in the task
    # Note: encode_line by default acts on Space-separated tokens if no line_tokenizer is passed
    # passing char_tokenize ensures we split into chars first.
    
    tokens = tgt_dict.encode_line(
        text, line_tokenizer=char_tokenize, add_if_not_exist=False, append_eos=False
    ).long()
    
    # 2. Add EOS to end (forming the 'target' sequence)
    # [c1, c2, ..., cn, EOS]
    eos_token = tgt_dict.eos()
    target_tokens = torch.cat([tokens, torch.tensor([eos_token])])
    target_tokens = target_tokens.unsqueeze(0).to(device) # [Batch=1, T]
    
    # 3. Construct prev_output_tokens (Input to model)
    # In valid_step:
    # sos_tokens = torch.full((tgt_tokens.shape[0], 1), model.decoder.dictionary.eos_index, ...)
    # tgt_tokens = torch.cat((sos_tokens, tgt_tokens), 1)
    
    # So input is [EOS, c1, ..., cn, EOS]
    sos_token = tgt_dict.eos() # Fairseq often uses EOS as BOS
    sos_tensor = torch.full((1, 1), sos_token, dtype=torch.long, device=device)
    
    prev_output_tokens = torch.cat([sos_tensor, target_tokens], dim=1)
    
    # 4. Call Model
    # model(prev_output_tokens, mode="segment")
    # This should return split_indices
    try:
        split_indices = model(prev_output_tokens, mode="segment")
    except TypeError:
        # Fallback if mode not supported (should not happen based on code analysis)
        st.error("Model does not support mode='segment'. Using manual Viterbi.")
        return get_viterbi_segmentation(model, task, text, device)
        
    # 5. Format Output
    # We can use task.split_text if available, or reproduce logic
    # task.split_text expects sample["target"]
    
    if hasattr(task, "split_text"):
        # Construct simplified sample
        sample = {
            "target": target_tokens, 
            "id": torch.tensor([0]),
            "nsentences": 1
        }
        try:
            split_texts = task.split_text(sample, split_indices)
            # split_texts is a list of strings with '|'
            if split_texts:
               result = split_texts[0]
               # Parse back to list of segments
               # result e.g. "seg1|seg2|seg3"
               # Remove </s> if present
               cleaned = result.replace("</s>", "").replace("<s>", "")
               segments = cleaned.split("|")
               return [s for s in segments if s] # filter empty
        except Exception as e:
            st.warning(f"task.split_text failed: {e}. Falling back to manual split.")
            pass
            
    # Manual split logic from task.split_text
    # split_indices is list of list of indices (ends of segments)
    # text reconstruction
    
    # Reconstruct text from target_tokens (to be safe about chars)
    # target_tokens: [c1, c2, ..., EOS]
    # We ignore EOS for text usually
    
    raw_text = ""
    target_list = target_tokens[0].cpu().tolist()
    # remove EOS from end if present
    if target_list[-1] == eos_token:
        valid_indices = target_list[:-1]
    else:
        valid_indices = target_list
        
    for idx in valid_indices:
        raw_text += tgt_dict.symbols[idx]
        
    raw_text = raw_text.replace("</s>", " ") # restore spaces
    
    # split_indices[0] contains indices in raw_text where segments end?
    # task logic: 
    # for counter, index in enumerate(split_indices[i]):
    #    text = text[:index + counter + 1] + "|" + text[index + counter + 1:]
    
    # This loop inserts '|'. counter accommodates for the shifting string length.
    
    current_text = raw_text
    indices = split_indices[0] # assuming batch 0
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
        
    # Indices seem to be end-indices.
    # Note: loop logic implies `index` is original index.
    
    segments = []
    last_idx = 0
    # indices are strictly increasing?
    # indices seem to be 0-based index of the last character of a segment.
    # e.g. "abc", segs "a", "bc". indices: 0, 2
    # text[0:1] -> "a"
    # text[1:3] -> "bc"
    
    for end_idx in indices:
        # end_idx is inclusive? Or exclusive?
        # task logic: text[:index + counter + 1] -> includes char at `index`
        # So `index` is the index of the character that ENDS the segment.
        # e.g. index 0 -> "a". text[:1]
        
        # We need to act on original string, so simpler than strings with pipes
        # slice raw_text[last_idx : end_idx + 1]
        seg = raw_text[last_idx : end_idx + 1]
        segments.append(seg)
        last_idx = end_idx + 1
        
    return segments

# --- Main App ---

def main():
    st.set_page_config(page_title="SSLM Visualizer", layout="wide")
    
    st.title("SSLM Segmentation Visualizer")
    
    st.sidebar.header("Configuration")

    # Hyperparameters
    # We might need these if the checkpoint doesn't store them fully or for overrides
    # But usually fairseq checkpoints have them.
    
    # Hugging Face Config
    st.sidebar.subheader("Hugging Face Source")
    hf_repo = st.sidebar.text_input("HF Repo ID", value=os.environ.get("HF_REPO_ID", ""))
    hf_token = st.sidebar.text_input("HF Token", value=os.environ.get("HF_TOKEN", ""), type="password")
    
    use_local = st.sidebar.checkbox("Use Local Checkpoints Only", value=not hf_repo)
    
    selected_ckpt_path = None
    
    if use_local:
        base_dir = os.path.abspath("checkpoints") # Default suggestion
        ckpt_dir = st.sidebar.text_input("Local Checkpoint Directory/File", value="")
        if ckpt_dir and os.path.isfile(ckpt_dir):
             selected_ckpt_path = ckpt_dir
        elif ckpt_dir and os.path.isdir(ckpt_dir):
             # List pt files
             files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
             if files:
                 selected_file = st.sidebar.selectbox("Select Checkpoint", files)
                 selected_ckpt_path = os.path.join(ckpt_dir, selected_file)
             else:
                 st.sidebar.warning("No .pt files found in directory.")
    else:
        if hf_repo:
            try:
                from huggingface_hub import HfApi, hf_hub_download
                api = HfApi(token=hf_token if hf_token else None)
                
                # List files
                files = api.list_repo_files(repo_id=hf_repo)
                ckpt_files = [f for f in files if f.endswith(".pt")]
                
                if not ckpt_files:
                    st.sidebar.warning(f"No .pt files found in {hf_repo}")
                else:
                    selected_ckpt_name = st.sidebar.selectbox("Select Remote Checkpoint", ckpt_files)
                    
                    if st.sidebar.button(f"Load {os.path.basename(selected_ckpt_name)}"):
                         with st.spinner("Downloading from Hugging Face..."):
                             try:
                                 selected_ckpt_path = hf_hub_download(
                                     repo_id=hf_repo,
                                     filename=selected_ckpt_name,
                                     token=hf_token if hf_token else None
                                 )
                                 st.session_state.current_ckpt_path = selected_ckpt_path
                             except Exception as e:
                                 st.error(f"Download failed: {e}")
                    
                    # Use cached path if available
                    if 'current_ckpt_path' in st.session_state and st.session_state.current_ckpt_path:
                         # Simple check to see if it matches current selection likely
                         # ideally we track what we downloaded.
                         # For now, just rely on the button or persistence.
                         if selected_ckpt_name in st.session_state.current_ckpt_path: 
                            selected_ckpt_path = st.session_state.current_ckpt_path
                                 
            except ImportError:
                st.error("huggingface_hub not installed. Please install it.")
            except Exception as e:
                st.error(f"Error fetching from HF: {e}")
        else:
            st.sidebar.info("Enter HF Repo ID to fetch checkpoints.")

    data_path = st.sidebar.text_input("Data Directory (containing dict.txt)", value="")
    
    st.sidebar.markdown("""
    **Note**: Ensure you have `fairseq` installed or available in the python path.
    """)

    if selected_ckpt_path:
        if not os.path.exists(selected_ckpt_path):
            st.error(f"Checkpoint not found at: {selected_ckpt_path}")
        else:
            # Load model
            # Re-load if changed
            if 'model' not in st.session_state or st.session_state.get('ckpt_path') != selected_ckpt_path:
                with st.spinner(f"Loading model from {os.path.basename(selected_ckpt_path)}..."):
                    model, task, cfg = load_model(selected_ckpt_path, data_path)
                    if model:
                        st.session_state.model = model
                        st.session_state.task = task
                        st.session_state.cfg = cfg
                        st.session_state.ckpt_path = selected_ckpt_path
                        st.success("Model loaded!")
            elif 'model' in st.session_state:
                 st.sidebar.success(f"Loaded: {os.path.basename(st.session_state.ckpt_path)}")
            
    # Input Area
    text = st.text_area("Input Text", "This is an example sentence.")
    
    if st.button("Visualize"):
        if 'model' in st.session_state and st.session_state.model:
            model = st.session_state.model
            task = st.session_state.task
            device = next(model.parameters()).device
            
            try:
                segments = get_model_segmentation(model, task, text, device)
                
                st.markdown("### Segmentation")
                
                # Display colored segments
                # Simple cycling colors
                colors = ["#FFD700", "#ADD8E6", "#90EE90", "#FFB6C1", "#FFA07A"]
                html = "<div style='line-height: 2.5; font-size: 18px;'>"
                for i, seg in enumerate(segments):
                    color = colors[i % len(colors)]
                    html += f"<span style='background-color: {color}; padding: 2px 4px; border-radius: 4px; margin-right: 2px; color: black;'>{seg}</span>"
                html += "</div>"
                
                st.markdown(html, unsafe_allow_html=True)
                
                st.write("Segments List:", segments)
                
            except Exception as e:
                st.error(f"Visualization failed: {e}")
                st.exception(e)
        else:
            st.warning("Please load a model first.")

if __name__ == "__main__":
    main()
