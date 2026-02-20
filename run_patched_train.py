import sys
import runpy
import torch
import os

# PyTorch 2.6+ defaults to weights_only=True which breaks fairseq Config objects.
# We monkey-patch torch.load to always disable weights_only for compatibility.
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# We also safely whitelist some known fairseq/omegaconf globals just in case
try:
    import omegaconf
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig, omegaconf.listconfig.ListConfig])
except Exception:
    pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_patched_train.py <script_to_run.py> [args...]")
        sys.exit(1)
        
    script_path = sys.argv[1]
    # Shift arguments so the inner script sees them correctly
    sys.argv = sys.argv[1:]
    
    # Prepend the fairseq directory to path if it's fairseq we are running
    import os
    if "fairseq_cli/train.py" in script_path:
        fairseq_dir = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))
        if fairseq_dir not in sys.path:
            sys.path.insert(0, fairseq_dir)
            
        from fairseq_cli.train import cli_main # type: ignore
        cli_main()
    else:
        # Fallback to runpy if they use this script for something else
        runpy.run_path(script_path, run_name="__main__")
