# SSLM Fertility Evaluation Script

This script evaluates the "fertility" (average number of subwords per segmented word) of Subword Segmental Language Model (SSLM) checkpoints.

It calculates the weighted average fertility on the validation set, providing insights into how the model's segmentation granularity evolves during training.

## Prerequisites
- **Python Environment**: Ensure `fairseq`, `torch`, `pandas`, `matplotlib`, and `huggingface_hub` are installed.
- **Data**: The script expects the validation data to be at `~/dataset/{LANG}/valid.{LANG}` by default, or you can provide a custom path.

## Usage

Run the script from this directory:

```bash
python evaluate_fertility.py \
    --repo-id <HF_REPO_ID> \
    --lang <SSLM_LANG_CODE> \
    --token <YOUR_HF_TOKEN>
```

### Arguments

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--repo-id` | Hugging Face repository ID containing checkpoints. | `sakethy/hin-sslm` |
| `--lang` | Language code. Used to find checkpoints in the repo and default data path. | `hin` |
| `--token` | Hugging Face API Token (optional if env var `HF_TOKEN` is set). | `hf_...` |
| `--data-path` | Optional explicit path to validation file. | `/path/to/valid.txt` |

### Example

To evaluate Hindi checkpoints:

```bash
export HF_TOKEN="hf_..."
python evaluate_fertility.py \
    --repo-id "sakethy/hin-sslm" \
    --lang hin
```

## Output
- **Console**: Prints Fertility score for each checkpoint.
- **CSV**: Saves results to `fertility_results.csv`.
- **Plot**: Saves `fertility_evolution.png` showing the trend over time.
