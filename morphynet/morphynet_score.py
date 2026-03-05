"""
MorphyNetScore: Evaluation of morphological alignment using the MorphyNet dataset.

MorphyNet provides two types of entries:
  - Inflectional: lemma → wordform with full multi-part segmentation (e.g. fogyasztó|i|hoz)
  - Derivational: lemma → wordform with a single affix (prefix or suffix)

Usage:
    from morphynet_score import MorphyNetScore
    ms = MorphyNetScore()
    infl = ms.load_inflectional('data/eng/eng.inflectional.v1.tsv')
    deriv = ms.load_derivational('data/eng/eng.derivational.v1.tsv')
    results = ms.evaluate(infl, lambda word: model_segment(word))
"""

import csv
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple


class MorphyNetScore:
    """
    Evaluates morphological segmentation against MorphyNet gold data.

    For inflectional data:
        The segmentation column uses '|' to separate morpheme parts.
        These parts are used directly as gold morphemes (can be 2 or more).
        No prefix/suffix distinction is made.
        Rows with segmentation='-' (irregular forms) are skipped.

    For derivational data:
        Each entry has a single affix (prefix or suffix).
        Gold morphemes are [affix, stem] (prefix) or [stem, affix] (suffix).
        affix_type is preserved so callers can filter by 'prefix' or 'suffix'.

    morph_eval() uses boundary-index matching — same logic as morphscore.py:
        - Find positions of predicted boundaries (cumulative character offsets)
        - Find gold boundary positions (same method)
        - Recall = fraction of gold boundaries hit
        - Precision = fraction of predicted boundaries that are gold
    """

    def __init__(
        self,
        exclude_single_tok: bool = True,
        exclude_single_morpheme: bool = True,
        exclude_numbers: bool = True,
    ):
        self.exclude_single_tok = exclude_single_tok
        self.exclude_single_morpheme = exclude_single_morpheme
        self.exclude_numbers = exclude_numbers

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_inflectional(self, tsv_path: str) -> List[Dict[str, Any]]:
        """
        Load inflectional entries from a MorphyNet TSV file.

        TSV format (no header): lemma<TAB>wordform<TAB>morphtag<TAB>segmentation
        Segmentation = '|'-separated morpheme parts, or '-' for irregular.

        Returns a list of dicts:
            {
                'wordform': str,
                'lemma':    str,
                'morphtag': str,
                'gold_morphemes': List[str],   # ordered morpheme parts
                'morph_type': 'inflectional',
            }
        Only entries where all parts concatenate to the exact wordform are kept.
        """
        entries = []
        seen = set()  # deduplicate (lemma, wordform) pairs

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 4:
                    continue
                lemma, wordform, morphtag, segmentation = row[0], row[1], row[2], row[3]

                # Skip irregular forms
                if segmentation.strip() == '-':
                    continue

                # Deduplicate
                key = (lemma, wordform)
                if key in seen:
                    continue
                seen.add(key)

                # Parse segmentation into morpheme parts
                parts = segmentation.split('|')
                if len(parts) < 2:
                    continue  # single-part segmentation: skip (no boundary)

                # Sanity check: concatenation must equal wordform
                if ''.join(parts) != wordform:
                    continue

                # Filter numbers
                if self.exclude_numbers and any(ch.isdigit() for ch in wordform):
                    continue

                entries.append({
                    'wordform': wordform,
                    'lemma': lemma,
                    'morphtag': morphtag,
                    'gold_morphemes': parts,
                    'morph_type': 'inflectional',
                })

        return entries

    def load_derivational(self, tsv_path: str) -> List[Dict[str, Any]]:
        """
        Load derivational entries from a MorphyNet TSV file.

        TSV format (no header):
            lemma<TAB>wordform<TAB>lemma_pos<TAB>wordform_pos<TAB>affix<TAB>affix_type

        Returns a list of dicts:
            {
                'wordform': str,
                'lemma':    str,
                'lemma_pos': str,
                'wordform_pos': str,
                'affix':    str,
                'affix_type': 'prefix' | 'suffix',
                'gold_morphemes': List[str],   # [affix, stem] or [stem, affix]
                'morph_type': 'derivational',
            }
        Only entries where the affix literally appears at the correct position in
        the wordform are kept.
        """
        entries = []
        seen = set()

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 6:
                    continue
                lemma, wordform, lemma_pos, wordform_pos, affix, affix_type = (
                    row[0], row[1], row[2], row[3], row[4], row[5]
                )

                affix_type = affix_type.strip().lower()
                if affix_type not in ('prefix', 'suffix'):
                    continue

                # Skip if affix not actually in wordform at the expected position
                if affix_type == 'prefix':
                    if not wordform.startswith(affix):
                        continue
                    stem = wordform[len(affix):]
                    gold_morphemes = [affix, stem]
                else:  # suffix
                    if not wordform.endswith(affix):
                        continue
                    stem = wordform[: len(wordform) - len(affix)]
                    gold_morphemes = [stem, affix]

                if not stem:
                    continue  # degenerate case

                # Filter numbers
                if self.exclude_numbers and any(ch.isdigit() for ch in wordform):
                    continue

                # Deduplicate (lemma, wordform)
                key = (lemma, wordform)
                if key in seen:
                    continue
                seen.add(key)

                entries.append({
                    'wordform': wordform,
                    'lemma': lemma,
                    'lemma_pos': lemma_pos,
                    'wordform_pos': wordform_pos,
                    'affix': affix,
                    'affix_type': affix_type,
                    'gold_morphemes': gold_morphemes,
                    'morph_type': 'derivational',
                })

        return entries

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def morph_eval(
        self, gold_morphemes: List[str], pred_segments: List[str]
    ) -> Tuple[float, float]:
        """
        Evaluate a single wordform segmentation.

        Uses boundary-index matching:
          - Gold boundaries: cumulative character positions between morphemes
          - Pred boundaries: cumulative character positions between tokens
          - Recall  = #{gold boundaries hit} / #{gold boundaries}
          - Precision = #{gold boundaries hit} / #{pred boundaries}

        Returns (recall, precision). Returns (nan, nan) when excluded by config.
        """
        # Single-token predictions
        if len(pred_segments) == 1:
            if self.exclude_single_tok:
                return (np.nan, np.nan)
            else:
                # Score: correct if wordform matches and gold has 1 morpheme too
                correct = (len(gold_morphemes) == 1 and
                           pred_segments[0] == gold_morphemes[0])
                s = 1.0 if correct else 0.0
                return (s, s)

        # Single-morpheme gold (no internal boundary)
        if len(gold_morphemes) == 1:
            if self.exclude_single_morpheme:
                return (np.nan, np.nan)
            else:
                correct = (len(pred_segments) == 1 and
                           pred_segments[0] == gold_morphemes[0])
                s = 1.0 if correct else 0.0
                return (s, s)

        # Compute gold boundary positions (all but last morpheme boundary)
        gold_boundaries = set()
        pos = 0
        for m in gold_morphemes[:-1]:
            pos += len(m)
            gold_boundaries.add(pos)

        # Compute pred boundary positions (all but last token boundary)
        pred_boundaries = []
        pos = 0
        for t in pred_segments[:-1]:
            pos += len(t)
            pred_boundaries.append(pos)

        if not pred_boundaries:
            # pred is effectively single token (empty splits), treat appropriately
            if self.exclude_single_tok:
                return (np.nan, np.nan)
            return (0.0, 0.0)

        n_gold = len(gold_boundaries)
        n_pred = len(pred_boundaries)
        n_hit = sum(1 for b in pred_boundaries if b in gold_boundaries)

        recall = n_hit / n_gold
        precision = n_hit / n_pred

        return (recall, precision)

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        entries: List[Dict[str, Any]],
        segment_fn: Callable,
        batch_size: int = 128
    ) -> Dict[str, Any]:
        """
        Evaluate a model over a list of MorphyNet entries.

        Args:
            entries:    Output of load_inflectional() or load_derivational()
            segment_fn: Function that takes a list of wordforms and returns a list
                        of segment string lists (e.g. [['micro', 'tome', 's']]).
            batch_size: Number of wordforms to pass to segment_fn at once.

        Returns a dict with:
            {
                'precision': float,
                'recall':    float,
                'f1':        float,
                'n_evaluated': int,     # samples actually scored (not nan)
                'n_total':     int,     # total entries
                'details': List[dict],  # per-sample info
            }
        """
        recalls = []
        precisions = []
        details = []

        for i in range(0, len(entries), batch_size):
            batch_entries = entries[i:i + batch_size]
            batch_wordforms = [entry['wordform'] for entry in batch_entries]

            try:
                # Ensure the function knows how to handle a list
                batch_preds = segment_fn(batch_wordforms)
            except Exception:
                batch_preds = [[wf] for wf in batch_wordforms]

            for entry, pred in zip(batch_entries, batch_preds):
                wordform = entry['wordform']
                gold = entry['gold_morphemes']

                r, p = self.morph_eval(gold, pred)

                if not np.isnan(r):
                    recalls.append(r)
                    precisions.append(p)

            details.append({
                'wordform': wordform,
                'ground_truth': ' '.join(gold),
                'predicted': ' '.join(pred),
                'recall': r,
                'precision': p,
                'morph_type': entry.get('morph_type', ''),
                'affix_type': entry.get('affix_type', 'n/a'),
            })

        n_eval = len(recalls)
        if n_eval == 0:
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'n_evaluated': 0, 'n_total': len(entries), 'details': details
            }

        mean_r = float(np.mean(recalls))
        mean_p = float(np.mean(precisions))
        f1 = (2 * mean_p * mean_r / (mean_p + mean_r)
              if (mean_p + mean_r) > 0 else 0.0)

        return {
            'precision': mean_p,
            'recall': mean_r,
            'f1': f1,
            'n_evaluated': n_eval,
            'n_total': len(entries),
            'details': details,
        }

    def evaluate_breakdown(
        self,
        infl_entries: Optional[List[Dict]] = None,
        deriv_entries: Optional[List[Dict]] = None,
        segment_fn: Optional[Callable] = None,
        morph_type: str = 'all',   # 'inflectional' | 'derivational' | 'all'
        affix_type: str = 'all',   # 'prefix' | 'suffix' | 'all'  (derivational only)
        batch_size: int = 128,
    ) -> Dict[str, Dict]:
        """
        Run evaluation with fine-grained breakdown:
          - inflectional (all, since no affix distinction)
          - derivational-suffix
          - derivational-prefix

        Returns a nested dict: { category_name: evaluate() result }
        """
        results = {}

        run_infl = morph_type in ('inflectional', 'all')
        run_deriv = morph_type in ('derivational', 'all')

        if run_infl and infl_entries is not None:
            results['inflectional'] = self.evaluate(infl_entries, segment_fn, batch_size=batch_size)

        if run_deriv and deriv_entries is not None:
            for at in ('suffix', 'prefix'):
                if affix_type not in (at, 'all'):
                    continue
                subset = [e for e in deriv_entries if e.get('affix_type') == at]
                results[f'derivational_{at}'] = self.evaluate(subset, segment_fn, batch_size=batch_size)

            if affix_type == 'all':
                results['derivational_all'] = self.evaluate(deriv_entries, segment_fn, batch_size=batch_size)

        return results


if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: morphynet_score.py <lang_data_dir>")
        print("  e.g. morphynet_score.py data/eng")
        sys.exit(1)

    data_dir = sys.argv[1]
    lang = os.path.basename(data_dir.rstrip('/'))

    ms = MorphyNetScore()

    infl_path = os.path.join(data_dir, f'{lang}.inflectional.v1.tsv')
    deriv_path = os.path.join(data_dir, f'{lang}.derivational.v1.tsv')

    if os.path.exists(infl_path):
        infl = ms.load_inflectional(infl_path)
        print(f'Inflectional entries: {len(infl)}')
        # Count per morpheme count
        from collections import Counter
        cnt = Counter(len(e['gold_morphemes']) for e in infl)
        print('Gold morpheme count distribution:', dict(sorted(cnt.items())))
        print('Sample:', infl[:3])
    else:
        print(f'Inflectional file not found: {infl_path}')

    if os.path.exists(deriv_path):
        deriv = ms.load_derivational(deriv_path)
        print(f'\nDerivational entries: {len(deriv)}')
        from collections import Counter
        cnt = Counter(e['affix_type'] for e in deriv)
        print('Affix type distribution:', dict(cnt))
        print('Sample:', deriv[:3])
    else:
        print(f'Derivational file not found: {deriv_path}')
