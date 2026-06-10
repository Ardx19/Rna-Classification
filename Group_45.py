# Generated from: rna_v_final_9_0.86.ipynb
# Converted at: 2026-04-16T17:21:34.564Z
# Runtime mode: standalone Python script

# # RNA Classification — v_final_5
# 
# **Five improvements over v_final_4 (0.84 LB):**
# 1. **Fuzzy matching** — 1–2 mismatch lookup against miRBase recovers taxonomy signal for unmatched sequences
# 2. **Frozen RNA-FM → PCA → LightGBM** — replaces broken fine-tuning with stable embedding extraction
# 3. **MFE z-score** — structural stability relative to shuffled controls (standard miRNA validation)
# 4. **Richer hairpin features** — precursor length, mature position, asymmetry
# 5. **Multi-level stacking** — proper ensemble with CatBoost + LightGBM + LogReg per block
# 
# ### Files required
# ```
# ./input/train.csv, ./input/test.csv, ./input/sample.csv
# ./input/mature.fa, ./input/hairpin.fa
# ./input/miRNA.dat (optional; if missing, v9 miRNA.dat features are zero-filled)
# ```
# ### GPU is optional (CPU works; RNA-FM embedding extraction is much slower on CPU)
# ### Expected: OOF ~0.85–0.88, LB ~0.86–0.90




# ── Cell 1: Install ───────────────────────────────────────────────────────────
import subprocess, sys, os
os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN'

def pip(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])

pip('ViennaRNA')
pip('multimolecule')
pip('transformers', 'torch')
pip('catboost')
print('Done')

# ── Cell 2: Imports & Config ──────────────────────────────────────────────────
import argparse
import re, warnings, copy, random
from collections import defaultdict
from pathlib import Path

import RNA
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='RNA classification pipeline runner')
    parser.add_argument('--input_path', default='./input', help='Directory containing train.csv, test.csv, sample.csv')
    parser.add_argument('--output_path', default='./output', help='Directory to write submission CSV files')
    parser.add_argument(
        '--all',
        choices=['on', 'off'],
        default='on',
        help='Save all submission variants (on) or only the best submission file (off)'
    )
    parser.add_argument(
        '--gpu',
        choices=['off', 'on', 'auto'],
        default='off',
        help='GPU mode: off (default), on (require CUDA if available), auto (use CUDA when available)'
    )
    return parser.parse_args()


ARGS = parse_args()
# Runtime config (kept in uppercase intentionally):
# DATA_DIR  <- --input_path
# FA_DIR    <- DATA_DIR
# OUTPUT_DIR <- --output_path (created automatically)
# SAVE_ALL  <- --all
# DEVICE    <- --gpu
DATA_DIR = Path(ARGS.input_path)
FA_DIR = DATA_DIR
OUTPUT_DIR = Path(ARGS.output_path)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_ALL = ARGS.all == 'on'

SEED     = 42
N_SPLITS = 5

if ARGS.gpu == 'on':
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        print('GPU requested (--gpu on) but CUDA is unavailable. Falling back to CPU.')
        DEVICE = torch.device('cpu')
elif ARGS.gpu == 'auto':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cpu')

np.random.seed(SEED)
random.seed(SEED)
print(f'Input dir: {DATA_DIR}')
print(f'FASTA dir: {FA_DIR}')
print(f'Output dir: {OUTPUT_DIR}')
print(f'Save all submissions: {SAVE_ALL}')
print(f'Device: {DEVICE}')

# ── Cell 3: Load data ─────────────────────────────────────────────────────────
train  = pd.read_csv(DATA_DIR / 'train.csv')
test   = pd.read_csv(DATA_DIR / 'test.csv')
sample = pd.read_csv(DATA_DIR / 'sample.csv')

y          = train['Label'].values.astype(np.int64)
train_seqs = train['Sequence'].str.upper().str.replace('T','U').tolist()
test_seqs  = test['Sequence'].str.upper().str.replace('T','U').tolist()

print(f'Train: {len(train)} | Test: {len(test)}')
print(f'Labels: {pd.Series(y).value_counts().to_dict()}')

# ── Cell 4: GroupKFold setup ──────────────────────────────────────────────────
def make_groups(seqs, threshold=0.5, k=6):
    kmers = [set(s[i:i+k] for i in range(len(s)-k+1)) for s in seqs]
    n = len(seqs)
    group = list(range(n))
    def find(x):
        while group[x] != x:
            group[x] = group[group[x]]
            x = group[x]
        return x
    def union(x, y): group[find(x)] = find(y)
    for i in range(n):
        for j in range(i+1, n):
            u = kmers[i] | kmers[j]
            if u and len(kmers[i] & kmers[j]) / len(u) >= threshold:
                union(i, j)
    return np.array([find(i) for i in range(n)])

print('Building similarity groups...')
groups = make_groups(train_seqs)
print(f'Groups: {len(np.unique(groups))} from {len(train_seqs)} sequences')
gkf = GroupKFold(n_splits=N_SPLITS)

# ── Cell 5: Parse FASTA files ─────────────────────────────────────────────────
def load_fasta(path):
    # Returns dict: name -> sequence  AND  sequence -> list of names
    name_to_seq = {}
    seq_to_names = defaultdict(list)
    cur_name, cur_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur_name:
                    s = ''.join(cur_seq).upper().replace('T','U')
                    name_to_seq[cur_name] = s
                    seq_to_names[s].append(cur_name)
                cur_name = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_name:
        s = ''.join(cur_seq).upper().replace('T','U')
        name_to_seq[cur_name] = s
        seq_to_names[s].append(cur_name)
    return name_to_seq, dict(seq_to_names)

print('Loading mature.fa...')
mature_name_to_seq, mature_seq_to_names = load_fasta(FA_DIR / 'mature.fa')
print(f'Mature: {len(mature_name_to_seq)} entries, {len(mature_seq_to_names)} unique seqs')

print('Loading hairpin.fa...')
hairpin_name_to_seq, _ = load_fasta(FA_DIR / 'hairpin.fa')
print(f'Hairpin: {len(hairpin_name_to_seq)} precursor entries')

# Build mature_name -> precursor_seq mapping
def mature_to_precursor_name(mature_name):
    # hsa-miR-21-5p -> hsa-mir-21
    name = re.sub(r'-(5p|3p)$', '', mature_name)
    name = name.replace('-miR-', '-mir-')
    name = name.replace('-miR*', '-mir-')
    return name

mature_to_hairpin = {}
for m_name in mature_name_to_seq:
    hp_name = mature_to_precursor_name(m_name)
    if hp_name in hairpin_name_to_seq:
        mature_to_hairpin[m_name] = hairpin_name_to_seq[hp_name]

train_hp_count = sum(
    1 for s in train_seqs
    for m in mature_seq_to_names.get(s, [])
    if m in mature_to_hairpin
)
print(f'Mature->hairpin mapping hits on train: {train_hp_count}')

# ── Cell 6: Fuzzy matching against miRBase ────────────────────────────────────
# KEY INSIGHT: exact matching misses sequences that are 1-2 nt away from known
# miRNAs. A sequence 1 mismatch from hsa-miR-21 should inherit miR-21's
# conservation profile. This recovers taxonomy signal for previously-zero seqs.
#
# Strategy: length-indexed Hamming lookup, max_dist=2, O(N*M) per length bucket.

print('Building fuzzy matching index...')

# Index miRBase mature sequences by length for fast lookup
mirbase_by_len = defaultdict(list)
for seq, names in mature_seq_to_names.items():
    mirbase_by_len[len(seq)].append((seq, names))

print(f'Length buckets: {sorted(mirbase_by_len.keys())[:10]}... ({len(mirbase_by_len)} total)')
print(f'Total unique seqs in miRBase: {len(mature_seq_to_names)}')

def hamming_distance(a, b):
    # Hamming distance between two equal-length strings.
    return sum(c1 != c2 for c1, c2 in zip(a, b))

def fuzzy_match(query, max_dist=2):
    # Find best miRBase match within max_dist mismatches.
    # Returns: (best_entries, best_distance, best_seq)
    qlen = len(query)
    best_dist    = max_dist + 1
    best_entries = []
    best_seq     = None

    # Same length: pure Hamming
    for ref_seq, ref_names in mirbase_by_len.get(qlen, []):
        d = hamming_distance(query, ref_seq)
        if d < best_dist:
            best_dist    = d
            best_entries = list(ref_names)
            best_seq     = ref_seq
        elif d == best_dist and d <= max_dist:
            best_entries.extend(ref_names)

    # ±1 length: try all single-character alignments (handles indels)
    for offset in [-1, 1]:
        ref_len = qlen + offset
        for ref_seq, ref_names in mirbase_by_len.get(ref_len, []):
            if offset == -1:
                # query is 1 longer — try deleting each position
                for pos in range(qlen):
                    trimmed = query[:pos] + query[pos+1:]
                    d = hamming_distance(trimmed, ref_seq) + 1  # +1 for the indel
                    if d < best_dist:
                        best_dist    = d
                        best_entries = list(ref_names)
                        best_seq     = ref_seq
                    elif d == best_dist and d <= max_dist:
                        best_entries.extend(ref_names)
            else:
                # query is 1 shorter — try inserting at each position
                for pos in range(ref_len):
                    trimmed = ref_seq[:pos] + ref_seq[pos+1:]
                    d = hamming_distance(query, trimmed) + 1
                    if d < best_dist:
                        best_dist    = d
                        best_entries = list(ref_names)
                        best_seq     = ref_seq
                    elif d == best_dist and d <= max_dist:
                        best_entries.extend(ref_names)

    if best_dist <= max_dist:
        return best_entries, best_dist, best_seq
    return [], max_dist + 1, None

# Run fuzzy matching for all train + test sequences
print('Fuzzy matching train sequences (may take 2-5 min)...')
train_fuzzy = []
for i, s in enumerate(train_seqs):
    exact = mature_seq_to_names.get(s, [])
    if exact:
        train_fuzzy.append((exact, 0, s))
    else:
        train_fuzzy.append(fuzzy_match(s, max_dist=2))
    if (i+1) % 500 == 0:
        print(f'  {i+1}/{len(train_seqs)}')

print('Fuzzy matching test sequences...')
test_fuzzy = []
for s in test_seqs:
    exact = mature_seq_to_names.get(s, [])
    if exact:
        test_fuzzy.append((exact, 0, s))
    else:
        test_fuzzy.append(fuzzy_match(s, max_dist=2))

# Coverage analysis
def coverage_report(fuzzy_results, label=''):
    exact  = sum(1 for e, d, _ in fuzzy_results if d == 0 and len(e) > 0)
    d1     = sum(1 for e, d, _ in fuzzy_results if d == 1)
    d2     = sum(1 for e, d, _ in fuzzy_results if d == 2)
    missed = sum(1 for e, d, _ in fuzzy_results if d > 2 or len(e) == 0)
    total  = len(fuzzy_results)
    print(f'{label}: exact={exact} d=1:{d1} d=2:{d2} missed={missed} total={total}')

coverage_report(train_fuzzy, 'Train')
coverage_report(test_fuzzy,  'Test')

# ── Cell 7: Hairpin structural features (improved) ───────────────────────────
# Now with: MFE z-score, precursor length, mature arm position, asymmetry
# Also: fuzzy matches get hairpin features from their best match

def fold_hairpin(hairpin_seq):
    # Fold a precursor and extract structural features.
    seq_str = hairpin_seq.replace('U', 'T')
    L = len(seq_str)
    if L < 10:
        return None

    structure, mfe = RNA.fold(seq_str)
    _, efe = RNA.pf_fold(seq_str)

    n_paired = structure.count('(')
    bp_frac  = 2 * n_paired / L

    loop_matches = re.findall(r'\((\\.+)\)', structure)
    loop_size = max((len(m) for m in loop_matches), default=0)

    stem_len = 0
    for ch in structure:
        if ch == '(':
            stem_len += 1
        else:
            break

    stem_region = structure[:structure.rfind('(') + 1]
    n_bulges = len(re.findall(r'\(\.*\.+\.*\(', stem_region))

    has_hairpin = int(loop_size > 0 and n_paired > 3)

    # NEW: Precursor length (real miRNAs have characteristic lengths ~60-120nt)
    precursor_length = L

    # NEW: Structure string features
    five_prime_overhang = 0
    for ch in structure:
        if ch == '.':
            five_prime_overhang += 1
        else:
            break

    three_prime_overhang = 0
    for ch in reversed(structure):
        if ch == '.':
            three_prime_overhang += 1
        else:
            break

    # NEW: Asymmetry of overhangs (real hairpins are relatively symmetric)
    asymmetry = abs(five_prime_overhang - three_prime_overhang) / (L + 1e-8)

    return [
        mfe, mfe / L, efe, bp_frac, stem_len, loop_size,
        has_hairpin, n_bulges, precursor_length,
        five_prime_overhang, three_prime_overhang, asymmetry,
    ]

STRUCT_FEAT_NAMES = [
    'mfe', 'mfe_per_nt', 'efe', 'bp_fraction', 'stem_length', 'loop_size',
    'has_hairpin', 'n_bulges', 'precursor_length',
    'five_prime_overhang', 'three_prime_overhang', 'asymmetry',
]

def mfe_zscore(hairpin_seq, n_shuffles=30):
    # Z-score of MFE vs shuffled controls.
    seq_str = hairpin_seq.replace('U', 'T')
    _, real_mfe = RNA.fold(seq_str)

    seq_list = list(seq_str)
    shuffle_mfes = []
    for _ in range(n_shuffles):
        shuffled = seq_list.copy()
        random.shuffle(shuffled)
        _, shuf_mfe = RNA.fold(''.join(shuffled))
        shuffle_mfes.append(shuf_mfe)

    mean_shuf = np.mean(shuffle_mfes)
    std_shuf  = np.std(shuffle_mfes)
    if std_shuf < 1e-8:
        return 0.0
    return (real_mfe - mean_shuf) / std_shuf

def get_hairpin_for_seq(query_seq, fuzzy_result):
    # Get precursor hairpin(s) using exact or fuzzy match.
    entries, dist, matched_seq = fuzzy_result

    hairpins = []
    for m_name in entries:
        if m_name in mature_to_hairpin:
            hairpins.append(mature_to_hairpin[m_name])

    # Also check via the matched sequence directly
    if matched_seq and matched_seq != query_seq:
        for m_name in mature_seq_to_names.get(matched_seq, []):
            if m_name in mature_to_hairpin:
                hp = mature_to_hairpin[m_name]
                if hp not in hairpins:
                    hairpins.append(hp)

    return hairpins

print('Extracting hairpin features with z-scores (~5-8 min)...')

def extract_struct_features(seqs, fuzzy_results):
    struct_rows = []
    zscore_rows = []
    n_found = 0

    for i, (s, fz) in enumerate(zip(seqs, fuzzy_results)):
        hairpins = get_hairpin_for_seq(s, fz)

        if hairpins:
            results = [fold_hairpin(hp) for hp in hairpins]
            results = [r for r in results if r is not None]
            if results:
                best = min(results, key=lambda r: r[0])
                struct_rows.append(best)

                # Z-score for the best hairpin
                best_hp = hairpins[results.index(best)] if len(results) == len(hairpins) else hairpins[0]
                zs = mfe_zscore(best_hp, n_shuffles=30)
                zscore_rows.append(zs)
                n_found += 1
                if (i+1) % 200 == 0:
                    print(f'  {i+1}/{len(seqs)} (found: {n_found})')
                continue

        struct_rows.append(None)
        zscore_rows.append(None)
        if (i+1) % 200 == 0:
            print(f'  {i+1}/{len(seqs)} (found: {n_found})')

    return struct_rows, zscore_rows, n_found

train_struct_raw, train_zscores, n_tr = extract_struct_features(train_seqs, train_fuzzy)
print(f'Train: {n_tr}/{len(train_seqs)} mapped')

test_struct_raw, test_zscores, n_te = extract_struct_features(test_seqs, test_fuzzy)
print(f'Test: {n_te}/{len(test_seqs)} mapped')

# Impute
found_struct = np.array([r for r in train_struct_raw if r is not None])
struct_means = found_struct.mean(axis=0)
found_zs     = [z for z in train_zscores if z is not None]
zscore_mean  = np.mean(found_zs) if found_zs else 0.0

def impute_struct(struct_rows, zscores, struct_means, zscore_mean):
    X_struct = np.array(
        [r if r is not None else struct_means.tolist() for r in struct_rows],
        dtype=np.float32
    )
    z_col = np.array(
        [z if z is not None else zscore_mean for z in zscores],
        dtype=np.float32
    ).reshape(-1, 1)
    return np.hstack([X_struct, z_col])

X_struct_train = impute_struct(train_struct_raw, train_zscores, struct_means, zscore_mean)
X_struct_test  = impute_struct(test_struct_raw,  test_zscores,  struct_means, zscore_mean)

STRUCT_FEAT_NAMES_FULL = STRUCT_FEAT_NAMES + ['mfe_zscore']
print(f'Structural feature matrix: {X_struct_train.shape}')

print('\nPer-feature AUCs:')
for i, name in enumerate(STRUCT_FEAT_NAMES_FULL):
    col = X_struct_train[:, i]
    if col.std() < 1e-9: continue
    auc = roc_auc_score(y, col)
    print(f'  {name:<25} {max(auc, 1-auc):.4f}')

# ── Cell 8: Taxonomy features with fuzzy matching ────────────────────────────
# Now uses fuzzy match results — sequences that didn't exact-match miRBase
# inherit the taxonomy profile of their closest match (at a penalty).

PRIMATES      = {'hsa','ptr','ppa','ggo','ppy','nle','mml','pon','sla','cja','ocu','aot','sge','cal','mne'}
RODENTS       = {'mmu','rno','cgr','dno','mdi','prd','cpb','ode','mze','gmo','hme','mus'}
OTHER_MAMMALS = {'bta','eca','ssc','cfa','oar','oan','mdo','lca','ttr','aja','chi','bmo','ovi','pcy','ste','mfu'}
BIRDS         = {'gga','tgu','mga','age','aca','pal','col','fal','tin','pet','mle','ana','str'}
FISH          = {'dre','fru','tni','xtr','ola','cca','cco','pca','ipu','lmi','pfo','obi','lch','pma','oni'}
INVERTEBRATES = {'cel','dme','aga','bmo','api','ame','hbr','cte','sky','aqu','nve','spu','lgi','bfl','ppc','dpu'}

def get_prefix(e):
    return e.split('-')[0] if '-' in e else ''

# Seed family size lookup
seed_counts = defaultdict(int)
for seq in mature_seq_to_names:
    if len(seq) >= 8:
        seed = seq[1:8]
        seed_counts[seed] += len(mature_seq_to_names[seq])

def build_taxon_features(seqs, fuzzy_results):
    # Build taxonomy features using exact OR fuzzy match entries.
    rows = []
    for s, (entries, edit_dist, matched_seq) in zip(seqs, fuzzy_results):
        prefixes = [get_prefix(e) for e in entries]

        n_prim   = sum(p in PRIMATES      for p in prefixes)
        n_rod    = sum(p in RODENTS       for p in prefixes)
        n_omam   = sum(p in OTHER_MAMMALS for p in prefixes)
        n_bird   = sum(p in BIRDS         for p in prefixes)
        n_fish   = sum(p in FISH          for p in prefixes)
        n_inv    = sum(p in INVERTEBRATES for p in prefixes)
        n_groups = sum([n_prim>0, n_rod>0, n_omam>0, n_bird>0, n_fish>0, n_inv>0])

        seed = s[1:8] if len(s) >= 8 else ''
        seed_fam = seed_counts.get(seed, 0)

        # NEW: fuzzy match features
        # edit_distance: 0 = exact, 1-2 = close, 3 = no match
        # is_exact_match: binary
        # fuzzy_seed_fam: seed family from matched sequence (if fuzzy)
        fuzzy_seed_fam = 0
        if matched_seq and len(matched_seq) >= 8:
            fuzzy_seed_fam = seed_counts.get(matched_seq[1:8], 0)

        rows.append([
            len(entries),
            n_prim, n_rod, n_omam, n_bird, n_fish, n_inv,
            n_groups,
            int(any(get_prefix(e)=='hsa' for e in entries)),
            int(any(get_prefix(e)=='mmu' for e in entries)),
            int(any(get_prefix(e)=='rno' for e in entries)),
            int(any('-5p' in e for e in entries)),
            int(any('-3p' in e for e in entries)),
            seed_fam,
            # NEW fuzzy features
            edit_dist,
            int(edit_dist == 0 and len(entries) > 0),   # is_exact_match
            max(seed_fam, fuzzy_seed_fam),               # best_seed_family
        ])
    return np.array(rows, dtype=np.float32)

TAXON_FEAT_NAMES = [
    'n_species', 'n_primate', 'n_rodent', 'n_other_mammal',
    'n_bird', 'n_fish', 'n_invertebrate', 'n_taxonomic_groups',
    'has_hsa', 'has_mmu', 'has_rno', 'has_5p', 'has_3p',
    'seed_family_size',
    'edit_distance', 'is_exact_match', 'best_seed_family',
]

print('Building taxonomy + fuzzy features...')
X_taxon_train_raw = build_taxon_features(train_seqs, train_fuzzy)
X_taxon_test_raw  = build_taxon_features(test_seqs,  test_fuzzy)

# Log-transform count columns
LOG_COLS = [0, 1, 2, 3, 4, 5, 6, 7, 13, 16]  # count-type columns
X_taxon_train = X_taxon_train_raw.copy()
X_taxon_test  = X_taxon_test_raw.copy()
X_taxon_train[:, LOG_COLS] = np.log1p(X_taxon_train_raw[:, LOG_COLS])
X_taxon_test[:, LOG_COLS]  = np.log1p(X_taxon_test_raw[:, LOG_COLS])

print(f'Taxonomy feature matrix: {X_taxon_train.shape}')
print('\nPer-feature AUCs:')
for i, name in enumerate(TAXON_FEAT_NAMES):
    col = X_taxon_train_raw[:, i]
    if col.std() < 1e-9: continue
    auc = roc_auc_score(y, col)
    print(f'  {name:<25} {max(auc, 1-auc):.4f}')

# Compare: how many sequences gained signal from fuzzy matching?
exact_only = sum(1 for e, d, _ in train_fuzzy if d == 0 and len(e) > 0)
fuzzy_gain = sum(1 for e, d, _ in train_fuzzy if 0 < d <= 2 and len(e) > 0)
print(f'\nExact matches: {exact_only}  |  Fuzzy recovered: {fuzzy_gain}  |  Total covered: {exact_only+fuzzy_gain}/{len(train_seqs)}')

# ── Cell 9.5: Add v9 features (miRNA.dat + triplet structure-sequence) ─────────
# Added exactly as requested: new feature blocks inserted after taxonomy/structure
# and before CV, then combined into model matrices.

def parse_mirna_dat(path):
    """Parse miRNA.dat to extract per-entry deep annotations."""
    entries = {}
    current_id = None
    current_data = {
        'n_references': 0,
        'n_experiments': 0,
        'total_reads': 0,
        'n_mature_products': 0,
        'mature_names': [],
        'description': '',
    }

    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')

            if line.startswith('ID   '):
                current_id = line[5:].split()[0].strip(';')
                current_data = {
                    'n_references': 0,
                    'n_experiments': 0,
                    'total_reads': 0,
                    'n_mature_products': 0,
                    'mature_names': [],
                    'description': '',
                }
            elif line.startswith('DE   '):
                current_data['description'] += line[5:].strip()
            elif line.startswith('RX   '):
                current_data['n_references'] += 1
            elif line.startswith('RC   '):
                nums = re.findall(r'(\d+)', line[5:])
                if nums:
                    current_data['total_reads'] += int(nums[0])
            elif line.startswith('DR   '):
                current_data['n_experiments'] += 1
            elif line.startswith('FT   '):
                ft_content = line[5:].strip()
                if 'miRNA' in ft_content or 'mature' in ft_content:
                    current_data['n_mature_products'] += 1
                product_match = re.search(r'/product="([^"]+)"', ft_content)
                if product_match:
                    current_data['mature_names'].append(product_match.group(1))
            elif line.startswith('//'):
                if current_id:
                    for mname in current_data['mature_names']:
                        entries[mname] = {
                            'precursor_id': current_id,
                            'n_references': current_data['n_references'],
                            'n_experiments': current_data['n_experiments'],
                            'total_reads': current_data['total_reads'],
                            'n_mature_products': current_data['n_mature_products'],
                        }
                    entries[current_id] = current_data.copy()

    return entries


def build_dat_features(seqs, fuzzy_results, dat_entries):
    rows = []
    for s, (fz_entries, edit_dist, matched_seq) in zip(seqs, fuzzy_results):
        names = fz_entries
        n_refs, n_exps, n_reads, n_products = [], [], [], []

        for name in names:
            if name in dat_entries:
                d = dat_entries[name]
                n_refs.append(d['n_references'])
                n_exps.append(d['n_experiments'])
                n_reads.append(d['total_reads'])
                n_products.append(d['n_mature_products'])

        if n_refs:
            rows.append([
                max(n_refs),
                np.mean(n_refs),
                max(n_exps),
                np.mean(n_exps),
                max(n_reads),
                np.mean(n_reads),
                max(n_products),
                len(n_refs),
                int(max(n_reads) > 0),
                int(max(n_refs) > 3),
            ])
        else:
            rows.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return np.array(rows, dtype=np.float32)


DAT_FEAT_NAMES = [
    'max_references', 'mean_references', 'max_experiments', 'mean_experiments',
    'max_reads', 'mean_reads', 'max_mature_products', 'n_dat_entries',
    'has_read_data', 'well_studied',
]


def compute_triplet_features(sequence, structure):
    if len(sequence) != len(structure):
        return None

    L = len(sequence)
    if L < 3:
        return None

    def struct_type(ch):
        return 'P' if ch in '()' else 'U'

    triplet_counts = defaultdict(int)
    for i in range(1, L - 1):
        left = struct_type(structure[i - 1])
        nuc = sequence[i].upper()
        right = struct_type(structure[i + 1])

        if nuc not in 'ACGU':
            nuc = 'A'

        triplet = f"{left}{nuc}{right}"
        triplet_counts[triplet] += 1

    triplet_labels = []
    for s1 in ['P', 'U']:
        for nuc in ['A', 'C', 'G', 'U']:
            for s2 in ['P', 'U']:
                triplet_labels.append(f"{s1}{nuc}{s2}")

    total = sum(triplet_counts.values())
    if total == 0:
        return np.zeros(len(triplet_labels), dtype=np.float32)

    return np.array([triplet_counts[t] / total for t in triplet_labels], dtype=np.float32)


def compute_extended_struct_features(hairpin_seq, mature_seq=None):
    seq_str = hairpin_seq.replace('U', 'T')
    L = len(seq_str)
    if L < 10:
        return None

    structure, mfe = RNA.fold(seq_str)
    hp_seq_rna = hairpin_seq.upper()

    triplet_vec = compute_triplet_features(hp_seq_rna, structure)
    if triplet_vec is None:
        return None

    mature_pos = -1
    mature_frac = 0.5
    if mature_seq and mature_seq in hp_seq_rna:
        mature_pos = hp_seq_rna.index(mature_seq)
        mature_frac = mature_pos / L

    is_5p_arm = int(mature_frac < 0.4) if mature_pos >= 0 else -1
    is_3p_arm = int(mature_frac > 0.6) if mature_pos >= 0 else -1

    mature_len = len(mature_seq) if mature_seq else 22
    len_ratio = mature_len / L

    if mature_pos >= 0 and mature_seq:
        mature_struct = structure[mature_pos:mature_pos + len(mature_seq)]
        mature_paired_frac = (mature_struct.count('(') + mature_struct.count(')')) / len(mature_struct)
    else:
        mature_paired_frac = 0.5

    extra = np.array([
        mature_frac, is_5p_arm, is_3p_arm, len_ratio, mature_paired_frac
    ], dtype=np.float32)

    return np.concatenate([triplet_vec, extra])


TRIPLET_FEAT_NAMES = [
    f"trip_{s1}{nuc}{s2}"
    for s1 in ['P', 'U'] for nuc in ['A', 'C', 'G', 'U'] for s2 in ['P', 'U']
] + [
    'mature_position_frac', 'is_5p_arm', 'is_3p_arm', 'mature_precursor_ratio', 'mature_paired_frac'
]


def extract_triplet_features(seqs, fuzzy_results, mature_to_hairpin, mature_seq_to_names):
    rows = []
    n_found = 0

    for i, (s, fz) in enumerate(zip(seqs, fuzzy_results)):
        entries, dist, matched_seq = fz

        hairpins = []
        for m_name in entries:
            if m_name in mature_to_hairpin:
                hairpins.append((mature_to_hairpin[m_name], s))

        if matched_seq and matched_seq != s:
            for m_name in mature_seq_to_names.get(matched_seq, []):
                if m_name in mature_to_hairpin:
                    hp = mature_to_hairpin[m_name]
                    if not any(h[0] == hp for h in hairpins):
                        hairpins.append((hp, matched_seq))

        if hairpins:
            results = []
            for hp_seq, mat_seq in hairpins:
                r = compute_extended_struct_features(hp_seq, mat_seq)
                if r is not None:
                    results.append(r)

            if results:
                rows.append(results[0])
                n_found += 1
                continue

        rows.append(None)

    found = np.array([r for r in rows if r is not None])
    means = found.mean(axis=0) if len(found) > 0 else np.zeros(21)

    X = np.array(
        [r if r is not None else means for r in rows],
        dtype=np.float32
    )

    print(f'Triplet features: {n_found}/{len(seqs)} mapped, {X.shape[1]} dims')
    return X


print('Building v9 extra features...')

mirna_dat_path = FA_DIR / 'miRNA.dat'
if mirna_dat_path.exists():
    print('Parsing miRNA.dat...')
    dat_entries = parse_mirna_dat(mirna_dat_path)
    print(f'Parsed {len(dat_entries)} entries from miRNA.dat')
    X_dat_train = build_dat_features(train_seqs, train_fuzzy, dat_entries)
    X_dat_test = build_dat_features(test_seqs, test_fuzzy, dat_entries)
else:
    print('miRNA.dat not found, filling miRNA.dat features with zeros')
    X_dat_train = np.zeros((len(train_seqs), len(DAT_FEAT_NAMES)), dtype=np.float32)
    X_dat_test = np.zeros((len(test_seqs), len(DAT_FEAT_NAMES)), dtype=np.float32)

print('Extracting triplet structure-sequence features...')
X_trip_train = extract_triplet_features(train_seqs, train_fuzzy, mature_to_hairpin, mature_seq_to_names)
X_trip_test = extract_triplet_features(test_seqs, test_fuzzy, mature_to_hairpin, mature_seq_to_names)

print('Per-feature AUCs (miRNA.dat):')
for i, name in enumerate(DAT_FEAT_NAMES):
    col = X_dat_train[:, i]
    if col.std() < 1e-9:
        continue
    auc = roc_auc_score(y, col)
    print(f'  {name:<25} {max(auc, 1 - auc):.4f}')

print('\nPer-feature AUCs (triplet):')
for i, name in enumerate(TRIPLET_FEAT_NAMES):
    col = X_trip_train[:, i]
    if col.std() < 1e-9:
        continue
    auc = roc_auc_score(y, col)
    if max(auc, 1 - auc) > 0.55:
        print(f'  {name:<25} {max(auc, 1 - auc):.4f}')

# Combined matrices used by downstream CV cells.
X_plus_train_raw = np.hstack([X_taxon_train_raw, X_struct_train, X_dat_train, X_trip_train])
X_plus_test_raw = np.hstack([X_taxon_test_raw, X_struct_test, X_dat_test, X_trip_test])
X_plus_train_log = np.hstack([X_taxon_train, X_struct_train, X_dat_train, X_trip_train])
X_plus_test_log = np.hstack([X_taxon_test, X_struct_test, X_dat_test, X_trip_test])

print(f'New total features before RNA-FM: {X_plus_train_raw.shape[1]}')

# ── Cell 10: CPU CV — compare feature blocks ──────────────────────────────────
# Tests now include the new v9 feature block (miRNA.dat + triplet features).

X_ts_train = np.hstack([X_taxon_train, X_struct_train])
X_ts_test = np.hstack([X_taxon_test, X_struct_test])

results = {}
feature_configs = {
    'taxon_only': (
        X_taxon_train,
        X_taxon_test,
        X_taxon_train_raw,
        X_taxon_test_raw,
    ),
    'taxon_struct': (
        X_ts_train,
        X_ts_test,
        np.hstack([X_taxon_train_raw, X_struct_train]),
        np.hstack([X_taxon_test_raw, X_struct_test]),
    ),
    'taxon_struct_v9plus': (
        X_plus_train_log,
        X_plus_test_log,
        X_plus_train_raw,
        X_plus_test_raw,
    ),
}

for name in feature_configs:
    results[name] = {'oof': np.zeros(len(y)), 'test': np.zeros(len(test_seqs)), 'fold_aucs': []}

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_taxon_train, y, groups)):
    y_tr, y_va = y[tr_idx], y[va_idx]

    for name, (Xtr_log, Xte_log, Xtr_raw, Xte_raw) in feature_configs.items():
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_log[tr_idx])
        Xva_s = sc.transform(Xtr_log[va_idx])
        Xte_s = sc.transform(Xte_log)

        lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
        lr.fit(Xtr_s, y_tr)
        oof_lr = lr.predict_proba(Xva_s)[:, 1]
        test_lr = lr.predict_proba(Xte_s)[:, 1]

        lgbm = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            reg_lambda=2.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(Xtr_raw[tr_idx], y_tr)
        oof_lgb = lgbm.predict_proba(Xtr_raw[va_idx])[:, 1]
        test_lgb = lgbm.predict_proba(Xte_raw)[:, 1]

        cb = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=SEED,
            verbose=0,
        )
        cb.fit(Xtr_raw[tr_idx], y_tr)
        oof_cb = cb.predict_proba(Xtr_raw[va_idx])[:, 1]
        test_cb = cb.predict_proba(Xte_raw)[:, 1]

        oof_p = (oof_lr + oof_lgb + oof_cb) / 3
        test_p = (test_lr + test_lgb + test_cb) / 3

        results[name]['oof'][va_idx] = oof_p
        results[name]['test'] += test_p / N_SPLITS
        results[name]['fold_aucs'].append(roc_auc_score(y_va, oof_p))

    print(
        f'Fold {fold+1}/{N_SPLITS} — '
        + ' | '.join(f'{n}={results[n]["fold_aucs"][-1]:.4f}' for n in results)
    )

print('\n── CPU CV Results ──')
for name in results:
    oof_auc = roc_auc_score(y, results[name]['oof'])
    fold_std = np.std(results[name]['fold_aucs'])
    print(f'  {name:<20} OOF={oof_auc:.4f}  std={fold_std:.4f}')

best_cpu_name = max(results, key=lambda n: roc_auc_score(y, results[n]['oof']))
oof_best_cpu = results[best_cpu_name]['oof']
test_best_cpu = results[best_cpu_name]['test']
print(f'\nBest CPU model: {best_cpu_name} (OOF={roc_auc_score(y, oof_best_cpu):.4f})')

# ── Cell 10: Save CPU-only submissions ────────────────────────────────────────
if SAVE_ALL:
    for name in results:
        sub = sample.copy()
        sub['Label'] = results[name]['test']
        sub.to_csv(OUTPUT_DIR / f'sub_{name}_cpu.csv', index=False)
        print(f'sub_{name}_cpu.csv  OOF={roc_auc_score(y, results[name]["oof"]):.4f}')

    print('\nCPU-only submissions saved. Submit best one before proceeding to GPU section.')
else:
    print('\nSkipping CPU-only submission file exports (--all off).')

# ---
# ## GPU section — Frozen RNA-FM embeddings (NOT fine-tuning)
# 
# **Key change from v_final_4:** Instead of fine-tuning (which overfits badly at 0.64 OOF
# with 1,529 samples and 640-dim model), we:
# 1. Extract frozen embeddings from the pretrained RNA-FM
# 2. PCA to 30–50 dimensions
# 3. Feed to LightGBM alongside taxonomy+structure features
# 
# This is more stable, faster (~2 min vs 40 min), and gives better signal extraction.


# ── Cell 11: Load RNA-FM & extract frozen embeddings ─────────────────────────
from multimolecule import RnaTokenizer, RnaFmModel

print('Loading RNA-FM (multimolecule/rnafm)...')
tokenizer  = RnaTokenizer.from_pretrained('multimolecule/rnafm')
rnafm_base = RnaFmModel.from_pretrained('multimolecule/rnafm')
rnafm_base = rnafm_base.to(DEVICE).eval()

EMB_DIM  = rnafm_base.config.hidden_size
N_LAYERS = rnafm_base.config.num_hidden_layers
print(f'RNA-FM: {N_LAYERS} layers, {EMB_DIM}-dim embeddings')

def tokenise_seqs(seqs, tokenizer, max_length=32):
    enc = tokenizer(seqs, return_tensors='pt', padding=True,
                    truncation=True, max_length=max_length)
    return enc['input_ids'], enc['attention_mask']

def extract_frozen_embeddings(seqs, tokenizer, model, batch_size=64):
    # Extract mean-pooled frozen embeddings from all layers.
    ids, mask = tokenise_seqs(seqs, tokenizer)
    all_embs = []

    for start in range(0, len(ids), batch_size):
        b_ids  = ids[start:start+batch_size].to(DEVICE)
        b_mask = mask[start:start+batch_size].to(DEVICE)

        with torch.no_grad():
            out = model(input_ids=b_ids, attention_mask=b_mask,
                       output_hidden_states=True)

        # Concatenate last 4 layers for richer representation
        hidden_states = out.hidden_states  # tuple of (B, L, D) per layer
        # Use last 4 layers
        stacked = torch.stack(hidden_states[-4:], dim=0)  # (4, B, L, D)
        combined = stacked.mean(dim=0)                     # (B, L, D) — average last 4

        # Mean pool over non-padding tokens
        m = b_mask.unsqueeze(-1).float()
        pooled = (combined * m).sum(1) / (m.sum(1) + 1e-8)  # (B, D)
        all_embs.append(pooled.cpu().numpy())

    return np.vstack(all_embs)

print('Extracting frozen embeddings (train)...')
emb_train = extract_frozen_embeddings(train_seqs, tokenizer, rnafm_base)
print(f'  Train embeddings: {emb_train.shape}')

print('Extracting frozen embeddings (test)...')
emb_test = extract_frozen_embeddings(test_seqs, tokenizer, rnafm_base)
print(f'  Test embeddings: {emb_test.shape}')

# PCA reduction: 640 -> 50 dims (regularises heavily)
N_PCA = 50
pca = PCA(n_components=N_PCA, random_state=SEED)
emb_train_pca = pca.fit_transform(emb_train).astype(np.float32)
emb_test_pca  = pca.transform(emb_test).astype(np.float32)
explained = pca.explained_variance_ratio_.sum()
print(f'PCA: {EMB_DIM} -> {N_PCA} dims, explained variance: {explained:.3f}')

# Quick sanity: AUC of first few PCs
print('\nTop PC AUCs:')
for i in range(min(10, N_PCA)):
    auc = roc_auc_score(y, emb_train_pca[:, i])
    print(f'  PC{i+1}: {max(auc, 1-auc):.4f}')

# ── Cell 14: CV with RNA-FM embeddings (frozen) ──────────────────────────────
# Uses the expanded v9-plus feature block + RNA-FM embeddings.

X_all_train = np.hstack([X_plus_train_log, emb_train_pca])
X_all_test = np.hstack([X_plus_test_log, emb_test_pca])
X_all_raw_train = np.hstack([X_plus_train_raw, emb_train_pca])
X_all_raw_test = np.hstack([X_plus_test_raw, emb_test_pca])

oof_emb = np.zeros(len(y))
test_emb = np.zeros(len(test_seqs))
fold_aucs_emb = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all_train, y, groups)):
    y_tr, y_va = y[tr_idx], y[va_idx]

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_all_train[tr_idx])
    Xva_s = sc.transform(X_all_train[va_idx])
    Xte_s = sc.transform(X_all_test)

    lr = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED)
    lr.fit(Xtr_s, y_tr)
    oof_lr = lr.predict_proba(Xva_s)[:, 1]
    test_lr = lr.predict_proba(Xte_s)[:, 1]

    lgbm = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=15,
        reg_lambda=5.0,
        colsample_bytree=0.6,
        subsample=0.8,
        subsample_freq=1,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(X_all_raw_train[tr_idx], y_tr)
    oof_lgb = lgbm.predict_proba(X_all_raw_train[va_idx])[:, 1]
    test_lgb = lgbm.predict_proba(X_all_raw_test)[:, 1]

    cb = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=5.0,
        subsample=0.8,
        bootstrap_type='Bernoulli',
        random_seed=SEED,
        verbose=0,
    )
    cb.fit(X_all_raw_train[tr_idx], y_tr)
    oof_cb = cb.predict_proba(X_all_raw_train[va_idx])[:, 1]
    test_cb = cb.predict_proba(X_all_raw_test)[:, 1]

    oof_p = (oof_lr + oof_lgb + oof_cb) / 3
    test_p = (test_lr + test_lgb + test_cb) / 3

    oof_emb[va_idx] = oof_p
    test_emb += test_p / N_SPLITS
    fold_aucs_emb.append(roc_auc_score(y_va, oof_p))
    print(f'  Fold {fold+1}: {fold_aucs_emb[-1]:.4f}')

emb_auc = roc_auc_score(y, oof_emb)
print(f'\nTaxon+struct+v9plus+RNA-FM(frozen) OOF={emb_auc:.4f}  std={np.std(fold_aucs_emb):.4f}')
print(f'vs best CPU model:                 OOF={roc_auc_score(y, oof_best_cpu):.4f}')
print(f'Embedding delta: {emb_auc - roc_auc_score(y, oof_best_cpu):+.4f}')

# ── Cell 13: Final multi-level stack ──────────────────────────────────────────
# Level 1 models (already computed):
#   A: taxon+struct (LR+LGBM+CB blend) -> oof_best_cpu
#   B: taxon+struct+RNA-FM-PCA (LR+LGBM+CB blend) -> oof_emb
#
# Level 2: LogReg meta-learner on OOF probabilities
# Also try: simple rank-average (more robust when calibrations differ)

meta_train = np.column_stack([oof_best_cpu, oof_emb])
meta_test  = np.column_stack([test_best_cpu, test_emb])

# Method 1: LogReg meta-stack
oof_meta   = np.zeros(len(y))
test_meta  = np.zeros(len(test_seqs))
fold_aucs_meta = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y, groups)):
    sc  = StandardScaler()
    Xtr = sc.fit_transform(meta_train[tr_idx])
    Xva = sc.transform(meta_train[va_idx])
    Xte = sc.transform(meta_test)

    meta_lr = LogisticRegression(C=0.2, max_iter=1000, random_state=SEED)
    meta_lr.fit(Xtr, y[tr_idx])

    oof_meta[va_idx] = meta_lr.predict_proba(Xva)[:, 1]
    test_meta       += meta_lr.predict_proba(Xte)[:, 1] / N_SPLITS
    fold_aucs_meta.append(roc_auc_score(y[va_idx], oof_meta[va_idx]))

meta_auc = roc_auc_score(y, oof_meta)

# Method 2: Rank average (non-parametric, robust)
from scipy.stats import rankdata
def rank_average(*arrays):
    ranks = [rankdata(a) / len(a) for a in arrays]
    return np.mean(ranks, axis=0)

oof_rank  = rank_average(oof_best_cpu, oof_emb)
test_rank = rank_average(test_best_cpu, test_emb)
rank_auc  = roc_auc_score(y, oof_rank)

# Method 3: Optimized weight blend (grid search on OOF)
best_blend_auc = 0
best_w = 0.5
for w in np.arange(0.1, 0.95, 0.05):
    blend = w * oof_best_cpu + (1 - w) * oof_emb
    blend_auc = roc_auc_score(y, blend)
    if blend_auc > best_blend_auc:
        best_blend_auc = blend_auc
        best_w = w

oof_wblend  = best_w * oof_best_cpu + (1 - best_w) * oof_emb
test_wblend = best_w * test_best_cpu + (1 - best_w) * test_emb

print('── Stacking Results ──')
print(f'  Component A ({best_cpu_name}):  OOF={roc_auc_score(y, oof_best_cpu):.4f}')
print(f'  Component B (w/ RNA-FM):        OOF={roc_auc_score(y, oof_emb):.4f}')
print(f'  LogReg meta-stack:              OOF={meta_auc:.4f}  std={np.std(fold_aucs_meta):.4f}')
print(f'  Rank average:                   OOF={rank_auc:.4f}')
print(f'  Weighted blend (w={best_w:.2f}):     OOF={best_blend_auc:.4f}')

# Auto-select best final model
final_options = {
    'meta_stack': (oof_meta, test_meta, meta_auc),
    'rank_avg':   (oof_rank, test_rank, rank_auc),
    'wblend':     (oof_wblend, test_wblend, best_blend_auc),
    'cpu_only':   (oof_best_cpu, test_best_cpu, roc_auc_score(y, oof_best_cpu)),
    'emb_full':   (oof_emb, test_emb, roc_auc_score(y, oof_emb)),
}
best_final = max(final_options, key=lambda k: final_options[k][2])
oof_final, test_final, final_auc = final_options[best_final]
print(f'\nBest overall: {best_final} (OOF={final_auc:.4f})')
print(f'Expected LB: ~{final_auc + 0.015:.3f} (based on historical +0.015 gap)')

# ── Cell 14: Save all submissions ─────────────────────────────────────────────
def save_sub(probs, name, oof=None):
    sub = sample.copy()
    sub['Label'] = probs
    sub.to_csv(OUTPUT_DIR / f'sub_{name}.csv', index=False)
    oof_str = f'OOF={roc_auc_score(y, oof):.4f}' if oof is not None else ''
    print(f'sub_{name}.csv  {oof_str}  prob=[{probs.min():.3f},{probs.max():.3f}]')

if SAVE_ALL:
    # Save all candidates
    save_sub(results['taxon_only']['test'],   'v5_taxon_only',   results['taxon_only']['oof'])
    save_sub(results['taxon_struct']['test'], 'v5_taxon_struct', results['taxon_struct']['oof'])
    save_sub(test_emb,    'v5_taxon_struct_rnafm', oof_emb)
    save_sub(test_meta,   'v5_meta_stack',   oof_meta)
    save_sub(test_rank,   'v5_rank_avg',     oof_rank)
    save_sub(test_wblend, 'v5_wblend',       oof_wblend)
    save_sub(test_final,  'v5_BEST',         oof_final)

    print(f'\n── Submit order ──')
    print(f'1. sub_v5_BEST.csv          ({best_final}, OOF={final_auc:.4f})')
    print(f'2. sub_v5_taxon_struct_rnafm.csv  (full features, OOF={roc_auc_score(y, oof_emb):.4f})')
    print(f'3. sub_v5_taxon_struct.csv  (CPU fallback, OOF={roc_auc_score(y, results["taxon_struct"]["oof"]):.4f})')
    print(f'\nv_final_4 benchmark: taxon_struct OOF=0.8137 LB=0.84')
else:
    # Save only one final submission file.
    save_sub(test_final, 'v5_BEST', oof_final)
    print(f'\nSaved only best submission: sub_v5_BEST.csv ({best_final}, OOF={final_auc:.4f})')

# ---
# ## Diagnostics — Error analysis
# 
# Understanding which sequences the model gets wrong helps guide next steps.


# ── Cell 15: Error analysis ───────────────────────────────────────────────────
# Which samples does the best model get most wrong?

errors = pd.DataFrame({
    'seq':   train_seqs,
    'label': y,
    'pred':  oof_final,
    'abs_error': np.abs(y - oof_final),
})

# Top false negatives (label=1 but predicted low)
fn = errors[errors['label'] == 1].nlargest(10, 'abs_error')
print('Top 10 false negatives (real miRNAs predicted as negative):')
for _, row in fn.iterrows():
    s = row['seq']
    entries, dist, _ = train_fuzzy[train_seqs.index(s)]
    n_ent = len(entries)
    print(f'  pred={row["pred"]:.3f}  dist={dist}  entries={n_ent}  {s[:30]}')

print()

# Top false positives (label=0 but predicted high)
fp = errors[errors['label'] == 0].nlargest(10, 'abs_error')
print('Top 10 false positives (non-miRNAs predicted as positive):')
for _, row in fp.iterrows():
    s = row['seq']
    entries, dist, _ = train_fuzzy[train_seqs.index(s)]
    n_ent = len(entries)
    print(f'  pred={row["pred"]:.3f}  dist={dist}  entries={n_ent}  {s[:30]}')

# Coverage vs accuracy
exact_mask = np.array([d == 0 and len(e) > 0 for e, d, _ in train_fuzzy])
fuzzy_mask = np.array([0 < d <= 2 and len(e) > 0 for e, d, _ in train_fuzzy])
nomatch    = ~exact_mask & ~fuzzy_mask

print(f'\nAccuracy by match type:')
for mask, name in [(exact_mask, 'Exact'), (fuzzy_mask, 'Fuzzy(1-2)'), (nomatch, 'No match')]:
    if mask.sum() > 0:
        subset_pred = (oof_final[mask] >= 0.5).astype(int)
        acc = (subset_pred == y[mask]).mean()
        auc = roc_auc_score(y[mask], oof_final[mask]) if len(np.unique(y[mask])) > 1 else float('nan')
        print(f'  {name:<12} n={mask.sum():4d}  acc={acc:.3f}  auc={auc:.3f}')