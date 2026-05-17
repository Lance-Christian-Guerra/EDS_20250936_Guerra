import sys
import os
import pandas as pd

EXPECTED_COLS = [
    "t","q1","q2","q3",
    "dq1","dq2","dq3",
    "I1","I2","I3",
    "eps21","eps22","eps31","eps32",
    "ddq1","ddq2","ddq3"
]

def detect_and_fix(input_path, out_path="data/dataset_original.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Read first row to inspect
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
    # Heuristic: if first line contains any alphabetic tokens matching expected names, treat as header
    first_tokens = [tok.strip().lower() for tok in first_line.replace('"','').split(',')]
    header_present = any(tok in EXPECTED_COLS for tok in first_tokens)
    if header_present:
        print("Header detected. Reading with header=0.")
        df = pd.read_csv(input_path, low_memory=False)
    else:
        print("No header detected. Reading with header=None and assigning expected column names.")
        df = pd.read_csv(input_path, header=None, low_memory=False)
        if df.shape[1] == len(EXPECTED_COLS):
            df.columns = EXPECTED_COLS
        else:
            # If column count differs, assign generic names and save a preview for inspection
            print(f"Column count mismatch: file has {df.shape[1]} columns but expected {len(EXPECTED_COLS)}.")
            generic_cols = [f"col_{i}" for i in range(df.shape[1])]
            df.columns = generic_cols
            preview_path = os.path.join(os.path.dirname(out_path), "preview_first_rows.csv")
            df.head(20).to_csv(preview_path, index=False)
            raise SystemExit(f"Saved preview to {preview_path}. Please inspect and provide a column mapping.")
    # Save corrected CSV
    df.to_csv(out_path, index=False)
    print(f"Saved corrected CSV to {out_path}. Preview:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_header.py /path/to/raw.csv")
        sys.exit(1)
    input_csv = sys.argv[1]
    detect_and_fix(input_csv)
