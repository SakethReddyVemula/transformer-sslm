
import sys
import os
from dataclasses import dataclass, field
from omegaconf import OmegaConf, II
from typing import Optional

# Mocking Fairseq classes
@dataclass
class FairseqDataclass:
    pass

@dataclass
class EncDecBaseConfig(FairseqDataclass):
    embed_dim: Optional[int] = field(
        default=512, metadata={"help": "embedding dimension"}
    )
    # ... other fields ignored for now

@dataclass
class DecoderConfig(EncDecBaseConfig):
    input_dim: int = II("model.decoder.embed_dim")
    output_dim: int = field(
        default=II("model.decoder.embed_dim"),
        metadata={
            "help": "decoder output dimension (extra linear layer if different from decoder embed dim)"
        },
    )

@dataclass
class SubwordSegmentalLanguageModelConfig(FairseqDataclass):
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder: DecoderConfig = DecoderConfig()

def safe_getattr(obj, name, default=None):
    return getattr(obj, name, default)

def main():
    # Simulate CLI args parsing
    cfg = SubwordSegmentalLanguageModelConfig()
    
    # Simulate CLI override: --decoder-embed-dim 128
    cfg.decoder_embed_dim = 128
    
    print(f"Initial cfg.decoder.embed_dim: {cfg.decoder.embed_dim}")
    
    # My fix in build_model:
    if hasattr(cfg, "decoder"):
        if safe_getattr(cfg, "decoder_embed_dim", None) is not None:
            cfg.decoder.embed_dim = cfg.decoder_embed_dim
            # Also sync input/output dims
            cfg.decoder.input_dim = cfg.decoder_embed_dim
            cfg.decoder.output_dim = cfg.decoder_embed_dim
            
            # CRITICAL FIX: Update top-level arg
            cfg.decoder_embed_dim = cfg.decoder.embed_dim
            
    print(f"After fix cfg.decoder.embed_dim: {cfg.decoder.embed_dim}")
    print(f"After fix cfg.decoder_embed_dim: {cfg.decoder_embed_dim}")
    
    # Mock SubwordSegmentalConfig
    @dataclass
    class SubwordSegmentalConfig(FairseqDataclass):
        decoder: DecoderConfig = DecoderConfig()
        
        @classmethod
        def from_namespace(cls, args):
            config = cls()
            # Logic from ssmt_config.py
            # It copies keys from args to config if they exist
            seen = set()
            
            # Simulate _copy_keys behavior for decoder
            # It looks for decoder_embed_dim in args
            if hasattr(args, "decoder_embed_dim"):
                 config.decoder.embed_dim = args.decoder_embed_dim
            
            return config

    ssmt_cfg = SubwordSegmentalConfig.from_namespace(cfg)
    
    print(f"SSMT Config decoder.embed_dim: {ssmt_cfg.decoder.embed_dim}")
    
    if ssmt_cfg.decoder.embed_dim == 128:
        print("SUCCESS: Config propagation works correctly.")
    else:
        print("FAILURE: Config propagation failed.")

if __name__ == "__main__":
    main()
