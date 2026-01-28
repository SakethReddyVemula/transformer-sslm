import sys
import os
import torch
from argparse import Namespace

# Add current directory to path to allow imports
# Insert at 0 to ensure local fairseq package is found before any namespace packages
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fairseq.models.transformer_sslm import SubwordSegmentalLanguageModel, SubwordSegmentalLanguageModelConfig
from fairseq.models.ssmt.ssmt_config import DecoderConfig
from fairseq.data import Dictionary

def estimate_params():
    # 1. Create Dummy Dictionaries
    # Char dictionary (approx 200 for Telugu + special)
    tgt_dict = Dictionary()
    # Add <eom> token which is required by the model
    tgt_dict.add_symbol("<eom>")
    
    # Add characters used in lexicon (w and digits)
    vocab_chars = "w0123456789"
    for c in vocab_chars:
        tgt_dict.add_symbol(c)
        
    # Fill the rest with dummy single characters to reach ~200
    for i in range(200):
        c = chr(ord('a') + i) 
        if c not in vocab_chars:
            tgt_dict.add_symbol(c)
    
    # Lexicon dictionary (10000 from script)
    tgt_lex = Dictionary()
    for i in range(10000):
        tgt_lex.add_symbol(f"w{i}")
        
    print(f"Char Vocab Size: {len(tgt_dict)}")
    print(f"Lexicon Vocab Size: {len(tgt_lex)}")

    # 2. Configure Model
    # Hyperparameters from santam_small.sh
    # DECODER_LAYERS=3
    # DECODER_EMBED_DIM=128
    # DECODER_FFN_EMBED_DIM=512
    # DECODER_ATTENTION_HEADS=4
    # MAX_SEG_LEN=5
    
    config = SubwordSegmentalLanguageModelConfig()
    config.decoder_layers = 3
    config.decoder_embed_dim = 128
    config.decoder_ffn_embed_dim = 512
    config.decoder_attention_heads = 4
    config.share_decoder_input_output_embed = True
    config.dropout = 0.1
    
    # Set DecoderConfig explicitly as build_model syncs them
    config.decoder = DecoderConfig()
    config.decoder.layers = 3
    config.decoder.embed_dim = 128
    config.decoder.ffn_embed_dim = 512
    config.decoder.attention_heads = 4
    config.decoder.max_seg_len = 5
    
    # Other defaults that might be relevant
    config.max_target_positions = 4096
    config.tokens_per_sample = 512
    
    # Mock Task
    class MockTask:
        def __init__(self, tgt_dict, tgt_lex):
            self.target_dictionary = tgt_dict
            self.target_lexicon = tgt_lex
            self.cfg = Namespace()
            self.cfg.max_seg_len = 5

    task = MockTask(tgt_dict, tgt_lex)

    # 3. Build Model
    model = SubwordSegmentalLanguageModel.build_model(config, task)

    # 4. Count Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture: {config.decoder_layers} layers, {config.decoder_embed_dim} embed dim, {config.decoder_ffn_embed_dim} FFN dim, {config.decoder_attention_heads} heads")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Breakdown
    print("\nBreakdown:")
    print(f"  Embeddings: {sum(p.numel() for p in model.decoder.embed_tokens.parameters()):,}")
    print(f"  Layers: {sum(p.numel() for p in model.decoder.layers.parameters()):,}")
    print(f"  Char Decoder: {sum(p.numel() for p in model.decoder.char_decoder.parameters()):,}")
    print(f"  Lex Decoder: {sum(p.numel() for p in model.decoder.lex_decoder.parameters()):,}")
    print(f"  Mixture Gate: {sum(p.numel() for p in model.decoder.mixture_gate_func.parameters()):,}")

if __name__ == "__main__":
    estimate_params()
