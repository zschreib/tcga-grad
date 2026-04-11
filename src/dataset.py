import os
import urllib.request
import pandas as pd
import GEOparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT_DIR, "data")

# ── Constants ─────────────────────────────────────────────────────────────────

GEO_ID = "GSE96058"
EXPRESSION_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/suppl/"
    "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz"
)
LABEL_COL = "pam50_subtype"
PAM50_CLASSES = ["LumA", "LumB", "Basal", "Her2", "Normal"]


# ── Download ──────────────────────────────────────────────────────────────────

def download_geo(geo_id=GEO_ID, save_dir=DATA_DIR):
    """Load GEO series from disk if available, otherwise download."""
    os.makedirs(save_dir, exist_ok=True)
    soft_file = os.path.join(save_dir, f"{geo_id}_family.soft.gz")

    if os.path.exists(soft_file):
        print(f"Loading GEO metadata from {soft_file}")
        gse = GEOparse.get_GEO(filepath=soft_file)
    else:
        print(f"Downloading GEO metadata for {geo_id}...")
        gse = GEOparse.get_GEO(geo=geo_id, destdir=save_dir)

    return gse


def download_expression_matrix(save_dir=DATA_DIR):
    """Download expression matrix if not already on disk."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "expression_matrix.csv.gz")

    if os.path.exists(save_path):
        print("Expression matrix already downloaded")
    else:
        print("Downloading expression matrix...")
        urllib.request.urlretrieve(EXPRESSION_URL, save_path)
        print("Done")

    return save_path


# ── Loading ───────────────────────────────────────────────────────────────────

def load_expression_matrix(save_dir=DATA_DIR):
    """Load expression matrix into a pandas DataFrame."""
    path = download_expression_matrix(save_dir)
    print("Loading expression matrix...")
    df = pd.read_csv(path, compression='gzip', index_col=0)
    print(f"Expression matrix shape: {df.shape}")
    return df


def extract_labels(gse):
    """Extract PAM50 subtype labels from GEO metadata."""
    labels = {}
    for sample_id, sample in gse.gsms.items():
        characteristics = sample.metadata.get('characteristics_ch1', [])
        for char in characteristics:
            if char.startswith('pam50 subtype:'):
                subtype = char.split(': ')[1].strip()
                labels[sample_id] = subtype
                break
    return pd.Series(labels, name=LABEL_COL)


# ── Alignment ─────────────────────────────────────────────────────────────────

def build_aligned_dataset(gse, df, labels):
    """Align expression matrix columns with PAM50 labels using sample titles."""

    # map expression matrix column names (F1, F2...) to GSM IDs
    title_to_gsm = {}
    for sample_id, sample in gse.gsms.items():
        title = sample.metadata.get('title', [None])[0]
        if title:
            title_to_gsm[title] = sample_id

    # align columns
    valid_columns = [col for col in df.columns if col in title_to_gsm]
    expression = df[valid_columns].T  # samples x genes
    gsm_ids = [title_to_gsm[col] for col in valid_columns]
    aligned_labels = labels.loc[gsm_ids]

    print(f"Aligned expression shape: {expression.shape}")
    print(f"Aligned labels shape:     {aligned_labels.shape}")

    return expression, aligned_labels


def encode_labels(labels):
    """Convert PAM50 string labels to integer class indices."""
    label_map = {subtype: idx for idx, subtype in enumerate(PAM50_CLASSES)}
    encoded = labels.map(label_map)
    print(f"\nLabel encoding: {label_map}")
    return encoded, label_map


# ── Main ──────────────────────────────────────────────────────────────────────

def load_dataset():
    """Full pipeline that returns aligned expression matrix and encoded labels."""
    gse = download_geo()
    df = load_expression_matrix()
    labels = extract_labels(gse)
    expression, aligned_labels = build_aligned_dataset(gse, df, labels)
    encoded_labels, label_map = encode_labels(aligned_labels)
    return expression, encoded_labels, label_map


if __name__ == "__main__":
    expression, labels, label_map = load_dataset()

    print("\n=== FINAL DATASET ===")
    print(f"Expression: {expression.shape}")
    print(f"Labels:     {labels.shape}")
    print(f"Classes:    {label_map}")
    print(f"\nLabel distribution:\n{labels.value_counts()}")
