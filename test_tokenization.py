
import nltk
import unicodedata

def is_alpha_extended(char):
    if char.isalpha():
        return True
    if unicodedata.category(char) in ['Mn', 'Mc']:
        return True
    return False

def is_seg_alpha(seg):
    return all(is_alpha_extended(c) for c in seg)

def tokenize_segs_original(line, max_seg_len, char_segs, non_alpha=False):
    # Split into all possible segments
    segs = []
    for n in range(1, max_seg_len+1):
        if n == 1 and not char_segs:
            continue

        chars = list(line)
        segs_n = nltk.ngrams(chars, n=n)
        segs_n = ["".join(seg) for seg in segs_n]

        if not non_alpha and n > 1:  # Discard segments with non-alphabetical characters
            segs_n = [seg for seg in segs_n if seg.isalpha() and len(seg) == n]
        else:
            segs_n = [seg for seg in segs_n if len(seg) == n]
        segs.extend(segs_n)
    return segs

def tokenize_segs_fixed(line, max_seg_len, char_segs, non_alpha=False):
    # Split into all possible segments
    segs = []
    for n in range(1, max_seg_len+1):
        if n == 1 and not char_segs:
            continue

        chars = list(line)
        segs_n = nltk.ngrams(chars, n=n)
        segs_n = ["".join(seg) for seg in segs_n]

        if not non_alpha and n > 1:  # Discard segments with non-alphabetical characters
            # Use extended alpha check
            segs_n = [seg for seg in segs_n if is_seg_alpha(seg) and len(seg) == n]
        else:
            segs_n = [seg for seg in segs_n if len(seg) == n]
        segs.extend(segs_n)
    return segs

def test():
    # Test with Hindi word "काम" (kaam) -> 'क' (ka), 'ा' (aa), 'म' (ma)
    # 'क' + 'ा' should be a valid segment of length 2
    word = "काम"
    print(f"Testing word: {word}")
    print("Original tokenize_segs:")
    segs_orig = tokenize_segs_original(word, max_seg_len=3, char_segs=True, non_alpha=False)
    print(segs_orig)
    
    print("\nFixed tokenize_segs:")
    segs_fixed = tokenize_segs_fixed(word, max_seg_len=3, char_segs=True, non_alpha=False)
    print(segs_fixed)

    # Check if 'क' + 'ा' is in segs
    seg_ka_aa = 'क' + 'ा'
    if seg_ka_aa in segs_orig:
        print(f"\n[FAIL] Original accepted {seg_ka_aa} (unexpected)")
    else:
        print(f"\n[PASS] Original rejected {seg_ka_aa} (expected failure)")

    if seg_ka_aa in segs_fixed:
        print(f"[PASS] Fixed accepted {seg_ka_aa}")
    else:
        print(f"[FAIL] Fixed rejected {seg_ka_aa}")

if __name__ == "__main__":
    test()
