import sys
import random

def split_corpus(file_path, lang):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Shuffle to ensure random distribution
        random.seed(42) 
        random.shuffle(lines)

        total = len(lines)
        train_end = int(total * 0.8)
        valid_end = int(total * 0.9)

        datasets = {
            f"train.{lang}": lines[:train_end],
            f"valid.{lang}": lines[train_end:valid_end],
            f"test.{lang}": lines[valid_end:]
        }

        for filename, data in datasets.items():
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(data)
            print(f"Created {filename} with {len(data)} lines.")

    except FileNotFoundError:
        print("Error: File not found.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <lang> <path_to_txt>")
    else:
        split_corpus(sys.argv[2], sys.argv[1])