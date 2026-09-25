# RASTRO-S

import os
import re
import math
import csv
import random
import statistics
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

ACC = "SRR10605873"
FASTA_PATH = f"/content/{ACC}.unitigs.fa"
OUT_CAND = "/content/branch_candidate_scores.csv"
OUT_DEC = "/content/branch_decisions_ngram.csv"

LOOKAHEAD_HOPS = 3
EXT_BP_LIMIT = 512
CONTEXT_TAIL = 256
KMER_K = 31
OVERLAP = KMER_K - 1
NGRAM_NS = [5, 6, 7, 10]
ALPHA_CTX = 1
ALPHA_UNI = 1

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def sh(cmd: str, check: bool = True):
    return subprocess.run(cmd, shell=True, check=check)

print("Installing zstd (if needed)…")
sh("apt-get -y update >/dev/null && apt-get -y install -y zstd >/dev/null", check=False)

if not Path(FASTA_PATH).exists():
    url = f"https://logan-pub.s3.amazonaws.com/u/{ACC}/{ACC}.unitigs.fa.zst"
    print("Downloading unitigs from Logan:", url)
    sh(f'curl -fsSL "{url}" | zstd -d -cq > "{FASTA_PATH}"')
else:
    print("Unitigs already present:", FASTA_PATH)

sh(f'ls -lh "{FASTA_PATH}"')
sh(f'head -n 2 "{FASTA_PATH}"')

_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def revcomp(s: str) -> str:
    return s.translate(_RC)[::-1]

_id_tail_re = re.compile(r"(\d+)$")
_link_re = re.compile(r"L:([+-]):([^:]+):([+-])")

@dataclass
class Neighbor:
    idx: int
    orient: str

@dataclass
class Unitig:
    idx: int
    name: str
    seq: str
    plus: List[Neighbor]
    minus: List[Neighbor]

@dataclass
class PathCandidate:
    branch_unitig: int
    branch_side: str
    start_neighbor: Neighbor
    nodes: List[Tuple[int, str]]
    ext_seq: str
    ext_bases: int

def oriented_seq(u: Unitig, orient: str) -> str:
    return u.seq if orient == "+" else revcomp(u.seq)

def neighbors_for_oriented(u: Unitig, orient: str) -> List[Neighbor]:
    return u.plus if orient == "+" else u.minus

def parse_logan_fasta(path: str) -> Dict[int, Unitig]:
    units: Dict[int, Unitig] = {}
    cur_name, cur_seq, cur_plus, cur_minus, cur_idx = None, [], [], [], None
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur_name is not None:
                    units[cur_idx] = Unitig(cur_idx, cur_name, "".join(cur_seq), cur_plus, cur_minus)
                header = line[1:].strip()
                cur_seq, cur_plus, cur_minus = [], [], []
                cur_name = header.split()[0]
                m = _id_tail_re.search(cur_name)
                if not m:
                    raise ValueError(f"Unitig name does not end in integer id: {cur_name}")
                cur_idx = int(m.group(1))
                for mlink in _link_re.finditer(header):
                    our, nid_token, nor = mlink.groups()
                    mid = _id_tail_re.search(nid_token)
                    if not mid:
                        continue
                    nid = int(mid.group(1))
                    (cur_plus if our == "+" else cur_minus).append(Neighbor(nid, nor))
            else:
                s = line.strip()
                if s:
                    cur_seq.append(s)
        if cur_name is not None:
            units[cur_idx] = Unitig(cur_idx, cur_name, "".join(cur_seq), cur_plus, cur_minus)
    return units

print("\nParsing FASTA…")
units = parse_logan_fasta(FASTA_PATH)
print("Parsed unitigs:", len(units))

def enumerate_candidates(
    units: Dict[int, Unitig],
    lookahead_hops: int,
    ext_bp_limit: int,
    overlap: int
) -> List[PathCandidate]:
    cands: List[PathCandidate] = []
    for uidx, u in units.items():
        for side, outs in (("+", u.plus), ("-", u.minus)):
            if len(outs) < 2:
                continue
            for nb in outs:
                nodes: List[Tuple[int, str]] = []
                ext_chunks: List[str] = []
                bases_added = 0
                hops = 0
                cur_idx, cur_or = nb.idx, nb.orient
                visited = {(uidx, side)}
                while True:
                    if (cur_idx, cur_or) in visited:
                        break
                    visited.add((cur_idx, cur_or))
                    cur = units.get(cur_idx)
                    if cur is None:
                        break
                    seq = oriented_seq(cur, cur_or)
                    nodes.append((cur.idx, cur_or))
                    if len(seq) <= overlap:
                        break
                    suffix = seq[overlap:]
                    take = min(len(suffix), max(0, ext_bp_limit - bases_added))
                    if take <= 0:
                        break
                    ext_chunks.append(suffix[:take])
                    bases_added += take
                    if bases_added >= ext_bp_limit:
                        break
                    outs2 = neighbors_for_oriented(cur, cur_or)
                    if len(outs2) != 1:
                        break
                    cur_idx, cur_or = outs2[0].idx, outs2[0].orient
                    hops += 1
                    if hops >= lookahead_hops:
                        break
                cands.append(PathCandidate(
                    branch_unitig=uidx,
                    branch_side=side,
                    start_neighbor=nb,
                    nodes=nodes,
                    ext_seq="".join(ext_chunks),
                    ext_bases=bases_added
                ))
    return cands

print("Enumerating candidates…")
cands = enumerate_candidates(units, LOOKAHEAD_HOPS, EXT_BP_LIMIT, OVERLAP)
print("Branch candidates:", len(cands))

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}

def train_ngram_from_unitigs(units: Dict[int, Unitig], n: int, alpha_ctx: int, alpha_uni: int):
    ctx_size = 4 ** n
    ctx_counts = np.full((ctx_size, 4), alpha_ctx, dtype=np.int64)
    ctx_obs = np.zeros(ctx_size, dtype=np.int64)
    uni_counts = np.full(4, alpha_uni, dtype=np.int64)
    mod = 4 ** (n - 1)
    for u in units.values():
        code = 0
        filled = 0
        for ch in u.seq.upper():
            if ch not in BASE_TO_INT:
                code = 0
                filled = 0
                continue
            b = BASE_TO_INT[ch]
            uni_counts[b] += 1
            if filled < n:
                code = code * 4 + b
                filled += 1
                continue
            ctx_counts[code, b] += 1
            ctx_obs[code] += 1
            code = (code % mod) * 4 + b
    uni_logp = np.log(uni_counts / uni_counts.sum())
    ctx_logp = np.log(ctx_counts / ctx_counts.sum(axis=1, keepdims=True))
    return ctx_logp, ctx_obs, uni_logp

def ngram_bpb(ctx_logp, ctx_obs, uni_logp, context: str, ext: str, n: int):
    mod = 4 ** (n - 1)
    code = 0
    filled = 0
    for ch in context.upper():
        if ch not in BASE_TO_INT:
            code = 0
            filled = 0
            continue
        b = BASE_TO_INT[ch]
        if filled < n:
            code = code * 4 + b
            filled += 1
        else:
            code = (code % mod) * 4 + b
    logps = []
    for ch in ext.upper():
        if ch not in BASE_TO_INT:
            code = 0
            filled = 0
            continue
        b = BASE_TO_INT[ch]
        lp = float(ctx_logp[code, b]) if (filled >= n and ctx_obs[code] > 0) else float(uni_logp[b])
        logps.append(lp)
        if filled < n:
            code = code * 4 + b
            filled += 1
        else:
            code = (code % mod) * 4 + b
    if not logps:
        return float("nan")
    return (-sum(logps) / len(logps)) / math.log(2)

print("Training RASTRO-S for n =", NGRAM_NS)
models = {}
for n in NGRAM_NS:
    print(f"  training n={n} …")
    models[n] = train_ngram_from_unitigs(units, n, ALPHA_CTX, ALPHA_UNI)
print("All models trained.")

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

print("Scoring candidates (RASTRO-S, all n)…")
rows = []
for cand in tqdm(cands, desc="Scoring RASTRO-S"):
    u = units[cand.branch_unitig]
    u_oriented = u.seq if cand.branch_side == "+" else revcomp(u.seq)
    context = u_oriented[-CONTEXT_TAIL:] if u_oriented else ""
    bu_name = u.name
    key = f"{bu_name}|{cand.branch_side}"
    path_nodes = "->".join(f"{nid}{ori}" for (nid, ori) in cand.nodes)
    start_neighbor = f"{cand.start_neighbor.idx}{cand.start_neighbor.orient}"
    for n in NGRAM_NS:
        ctx_logp, ctx_obs, uni_logp = models[n]
        ng_bpb = ngram_bpb(ctx_logp, ctx_obs, uni_logp, context, cand.ext_seq, n)
        rows.append({
            "branch_key": key,
            "branch_unitig": bu_name,
            "branch_side": cand.branch_side,
            "start_neighbor": start_neighbor,
            "path_nodes": path_nodes,
            "ext_bases": cand.ext_bases,
            "n": n,
            "RASTRO_bpb": ng_bpb,
        })

df = pd.DataFrame(rows)
df = df.sort_values(
    ["branch_unitig", "branch_side", "n", "RASTRO_bpb", "path_nodes"],
    na_position="last"
).reset_index(drop=True)
df.to_csv(OUT_CAND, index=False)
print("Wrote:", OUT_CAND, "| rows:", len(df))

dec_rows = []
for (bu, bs, n), g in df.groupby(["branch_unitig", "branch_side", "n"], sort=False):
    key = f"{bu}|{bs}"
    g2 = g[np.isfinite(g["RASTRO_bpb"].to_numpy())].sort_values(["RASTRO_bpb", "path_nodes"])
    nn = int(len(g2))
    if nn == 0:
        dec_rows.append({
            "branch_key": key, "branch_unitig": bu, "branch_side": bs, "n": n,
            "n_candidates": int(len(g)),
            "RASTRO_choice": None,
            "RASTRO_bpb_best": float("nan"),
            "RASTRO_bpb_second": float("nan"),
            "RASTRO_margin": float("nan"),
        })
        continue
    best_choice = g2.iloc[0]["path_nodes"]
    best_bpb = float(g2.iloc[0]["RASTRO_bpb"])
    second_bpb = float(g2.iloc[1]["RASTRO_bpb"]) if nn >= 2 else float("nan")
    margin = (second_bpb - best_bpb) if math.isfinite(second_bpb) else float("nan")
    dec_rows.append({
        "branch_key": key, "branch_unitig": bu, "branch_side": bs, "n": n,
        "n_candidates": int(len(g)),
        "RASTRO_choice": best_choice,
        "RASTRO_bpb_best": best_bpb,
        "RASTRO_bpb_second": second_bpb,
        "RASTRO_margin": margin,
    })

dec_df = pd.DataFrame(dec_rows).sort_values(["branch_unitig", "branch_side", "n"]).reset_index(drop=True)
dec_df.to_csv(OUT_DEC, index=False)
print("Wrote:", OUT_DEC, "| rows:", len(dec_df))

# RASTRO-S evaluation

CAND_CSV = "/content/branch_candidate_scores.csv"
DEC_NG_FULL = "/content/branch_decisions_ngram.csv"
UNITIGS_FA = f"/content/{ACC}.unitigs.fa"
READS_R1 = f"/content/{ACC}_1.fastq.gz"
READS_R2 = f"/content/{ACC}_2.fastq.gz"
READS_SUBSET = 250000
K = 31
COUNT_CANONICAL = True
EXT_LIMIT = 512
CONTEXT_TAIL_EVAL = 200
OVERLAP_EVAL = 30
N_SAMPLE = 20000
SEED = 13
SUB_R1 = "/content/reads_subset_R1.fastq"
SUB_R2 = "/content/reads_subset_R2.fastq"
JF_DB = "/content/reads_k31.jf"
SAMPLE_KEYS_TXT = "/content/ablation_sample_keys.txt"
READ_LABELS_CSV = "/content/ablation_read_labels.csv"
DEC_NG_SAMPLE = "/content/ablation_decisions_ngram_sample.csv"
EVAL_OUT = "/content/ablation_eval_ngram.csv"

def curl_text(url: str):
    p = subprocess.run(f'curl -sS -L -H "User-Agent: Mozilla/5.0" "{url}"', shell=True, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[:300])
    return p.stdout

def url_exists(url: str) -> bool:
    p = subprocess.run(f'curl -sI -L -H "User-Agent: Mozilla/5.0" "{url}" | head -n 1',
                       shell=True, text=True, capture_output=True)
    line = (p.stdout or "").upper()
    return ("200" in line) or ("206" in line)

def sra_ftp_guess_base(acc: str):
    prefix6 = acc[:6]
    digits = acc[3:]
    if len(digits) <= 7:
        mid = f"{int(digits[-1]):03d}"
    else:
        mid = f"{int(digits[-2:]):03d}"
    return f"https://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix6}/{mid}/{acc}/"

def ensure_reads(acc: str):
    if Path(READS_R1).exists() and Path(READS_R2).exists() and Path(READS_R1).stat().st_size > 0 and Path(READS_R2).stat().st_size > 0:
        print("Reads found in /content.")
        return
    print("Reads missing; trying ENA filereport API…")
    api = ("https://www.ebi.ac.uk/ena/portal/api/filereport"
           f"?accession={acc}&result=read_run"
           "&fields=run_accession,fastq_http,fastq_ftp,submitted_http,submitted_ftp"
           "&format=tsv")
    links = ""
    try:
        txt = curl_text(api)
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        if len(lines) >= 2:
            header = lines[0].split("\t")
            row = dict(zip(header, lines[1].split("\t")))
            for kf in ("fastq_http", "fastq_ftp", "submitted_http", "submitted_ftp"):
                v = (row.get(kf) or "").strip()
                if v:
                    links = v
                    break
    except Exception as e:
        print("ENA API failed; will try FTP-guess. Reason:", repr(e))
    def to_https(p: str) -> str:
        if p.startswith("ftp://"):
            p = p[len("ftp://"):]
        if p.startswith("http://"):
            p = "https://" + p[len("http://"):]
        if p.startswith("https://"):
            return p
        return "https://" + p
    if links:
        parts = [to_https(x.strip()) for x in links.split(";") if x.strip()]
        r1_url = next((u for u in parts if u.endswith("_1.fastq.gz") or u.endswith("_1.fq.gz")), None)
        r2_url = next((u for u in parts if u.endswith("_2.fastq.gz") or u.endswith("_2.fq.gz")), None)
        if r1_url and r2_url:
            sh(f'curl -L --retry 8 --retry-delay 2 -o "{READS_R1}" "{r1_url}"')
            sh(f'curl -L --retry 8 --retry-delay 2 -o "{READS_R2}" "{r2_url}"')
            if Path(READS_R1).stat().st_size > 0 and Path(READS_R2).stat().st_size > 0:
                print("Reads ready:", READS_R1, READS_R2)
                return
    print("Falling back to ENA SRA FTP structure guess…")
    base = sra_ftp_guess_base(acc)
    u1 = base + f"{acc}_1.fastq.gz"
    u2 = base + f"{acc}_2.fastq.gz"
    if url_exists(u1) and url_exists(u2):
        print("Found paired FASTQs at:", base)
        sh(f'curl -L --retry 8 --retry-delay 2 -o "{READS_R1}" "{u1}"')
        sh(f'curl -L --retry 8 --retry-delay 2 -o "{READS_R2}" "{u2}"')
        print("Reads ready:", READS_R1, READS_R2)
        return
    raise RuntimeError("Could not locate reads via ENA API or FTP guess.")

sh("apt-get -y update >/dev/null && apt-get -y install -y curl jellyfish >/dev/null", check=False)

assert Path(CAND_CSV).exists(), f"Missing {CAND_CSV}"
assert Path(DEC_NG_FULL).exists(), f"Missing {DEC_NG_FULL}"
assert Path(UNITIGS_FA).exists(), f"Missing {UNITIGS_FA}"

ensure_reads(ACC)
print("Reads ready. Paired:", True)

print(f"\nPreparing read subset: first {READS_SUBSET} reads from each mate…")
lines = READS_SUBSET * 4
if not Path(SUB_R1).exists():
    sh(f'gzip -dc "{READS_R1}" | head -n {lines} > "{SUB_R1}"')
if not Path(SUB_R2).exists():
    sh(f'gzip -dc "{READS_R2}" | head -n {lines} > "{SUB_R2}"')
print("Subset written.")

if not Path(JF_DB).exists():
    print("\nCounting read k-mers with jellyfish…")
    sh("rm -f /content/reads_k31.jf /content/reads_k31.jf_*", check=False)
    ret = subprocess.run(
        f'jellyfish count -m {K} -s 100M -t 2 {"-C" if COUNT_CANONICAL else ""} -o "{JF_DB}" "{SUB_R1}" "{SUB_R2}"',
        shell=True
    )
    if ret.returncode != 0:
        ret2 = subprocess.run(
            f'jellyfish count -m {K} -s 300M -t 2 {"-C" if COUNT_CANONICAL else ""} -o "{JF_DB}" "{SUB_R1}" "{SUB_R2}"',
            shell=True
        )
        if ret2.returncode != 0:
            raise RuntimeError("jellyfish count failed.")
print("Jellyfish DB ready:", JF_DB)

rng = random.Random(SEED)
reservoir = []
seen = 0
with open(DEC_NG_FULL, "r", newline="") as f:
    rd = csv.DictReader(f)
    for row in rd:
        seen += 1
        if len(reservoir) < N_SAMPLE:
            reservoir.append(row)
        else:
            j = rng.randrange(seen)
            if j < N_SAMPLE:
                reservoir[j] = row

sample_keys = [r["branch_key"] for r in reservoir]
Path(SAMPLE_KEYS_TXT).write_text("\n".join(sample_keys) + "\n")
print(f"\nSampled {len(sample_keys)} branch-sides (from {seen}).")
print("Saved:", SAMPLE_KEYS_TXT)

with open(DEC_NG_SAMPLE, "w", newline="") as out:
    cols = ["branch_key", "branch_unitig", "branch_side", "n_candidates", "RASTRO_choice", "RASTRO_bpb_best", "RASTRO_bpb_second", "RASTRO_margin"]
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in reservoir:
        w.writerow({c: r.get(c, "") for c in cols})
print("Saved:", DEC_NG_SAMPLE)

sample_set = set(sample_keys)
cand_paths = defaultdict(list)
with open(CAND_CSV, "r", newline="") as f:
    rd = csv.DictReader(f)
    for row in rd:
        k = row["branch_key"]
        if k not in sample_set:
            continue
        pn = (row.get("path_nodes") or "").strip()
        if pn:
            cand_paths[k].append(pn)

eval_keys = [k for k in sample_keys if len(cand_paths.get(k, [])) >= 2]
print("Keys to evaluate (>=2 candidates):", len(eval_keys))

_id_tail_re = re.compile(r"(\d+)$")
_link_node_re = re.compile(r"(\d+)([+-])$")

def tail_id(name: str) -> int:
    m = _id_tail_re.search(name)
    if not m:
        raise ValueError(f"No numeric tail in {name}")
    return int(m.group(1))

def parse_path_nodes(path_nodes: str):
    out = []
    for tok in (path_nodes or "").split("->"):
        tok = tok.strip()
        if not tok:
            continue
        m = _link_node_re.fullmatch(tok)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out

needed_ids = set()
branch_id = {}
for kkey in eval_keys:
    bu = kkey.split("|")[0]
    bid = tail_id(bu)
    branch_id[kkey] = bid
    needed_ids.add(bid)
    for pn in cand_paths[kkey]:
        for nid, _ in parse_path_nodes(pn):
            needed_ids.add(nid)

print("Unitig IDs needed:", len(needed_ids))

unitig_seq = {}
cur_id = None
cur_keep = False
buf = []
with open(UNITIGS_FA, "r") as fh:
    for line in fh:
        if line.startswith(">"):
            if cur_id is not None and cur_keep:
                unitig_seq[cur_id] = "".join(buf)
            hdr = line[1:].strip().split()[0]
            m = _id_tail_re.search(hdr)
            cur_id = int(m.group(1)) if m else None
            cur_keep = (cur_id in needed_ids) if cur_id is not None else False
            buf = []
        else:
            if cur_keep:
                s = line.strip()
                if s:
                    buf.append(s)
if cur_id is not None and cur_keep:
    unitig_seq[cur_id] = "".join(buf)

print("Loaded unitigs:", len(unitig_seq), "/", len(needed_ids))

ALPH = set("ACGT")

def canonical_kmer(km: str) -> str:
    rc = revcomp(km)
    return km if km <= rc else rc

def build_ext_from_path(path_nodes: str, ext_limit: int) -> str:
    nodes = parse_path_nodes(path_nodes)
    out = []
    total = 0
    for nid, orient in nodes:
        seq = unitig_seq.get(nid)
        if not seq:
            break
        seq_or = seq if orient == "+" else revcomp(seq)
        if len(seq_or) <= OVERLAP_EVAL:
            break
        suffix = seq_or[OVERLAP_EVAL:]
        take = min(len(suffix), ext_limit - total)
        if take <= 0:
            break
        out.append(suffix[:take])
        total += take
        if total >= ext_limit:
            break
    return "".join(out)

def iter_boundary_kmers(ctx: str, ext: str, k: int):
    if not ctx or not ext:
        return
    comb = (ctx + ext).upper()
    ext_start = len(ctx)
    start_i = max(0, ext_start - (k - 1))
    end_i = min(ext_start - 1, len(comb) - k)
    for i in range(start_i, end_i + 1):
        if i + k <= ext_start:
            continue
        km = comb[i:i+k]
        if "N" in km or any(ch not in ALPH for ch in km):
            continue
        yield canonical_kmer(km) if COUNT_CANONICAL else km

branch_info = {}
all_boundary = set()
for kkey in eval_keys:
    bid = branch_id[kkey]
    bseq = unitig_seq.get(bid, "")
    if not bseq:
        continue
    bs = kkey.split("|")[1]
    bseq_or = (bseq if bs == "+" else revcomp(bseq)).upper()
    ctx = bseq_or[-min(len(bseq_or), CONTEXT_TAIL_EVAL):]
    cand_list, bsets = [], []
    seen_pn = set()
    for pn in cand_paths[kkey]:
        if pn in seen_pn:
            continue
        seen_pn.add(pn)
        ext = build_ext_from_path(pn, EXT_LIMIT).upper()
        bset = set(iter_boundary_kmers(ctx, ext, K))
        cand_list.append(pn)
        bsets.append(bset)
        all_boundary.update(bset)
    if len(cand_list) < 2:
        continue
    branch_info[kkey] = {"candidates": cand_list, "bnd": bsets}

print("Total unique boundary kmers:", len(all_boundary))

QFA = "/content/disc_kmers.fa"
QOUT = "/content/disc_kmers.query.txt"
with open(QFA, "w") as f:
    for i, km in enumerate(all_boundary):
        f.write(f">k{i}\n{km}\n")

sh(f'jellyfish query "{JF_DB}" -s "{QFA}" > "{QOUT}"')

kmer_count = {}
with open(QOUT, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            kmer_count[parts[0]] = int(float(parts[1]))

def score_set(kmers):
    s = 0
    for km in kmers:
        s += kmer_count.get(km, 0)
    return s

labels = []
for kkey, info in branch_info.items():
    scores = [(score_set(bset), pn) for bset, pn in zip(info["bnd"], info["candidates"])]
    scores.sort(key=lambda x: x[0], reverse=True)
    best_sc, best_pn = scores[0]
    second_sc = scores[1][0] if len(scores) > 1 else 0
    tie = (best_sc == second_sc)
    unique = (best_sc > 0 and not tie)
    ratio = (best_sc / second_sc) if (second_sc > 0) else (float("inf") if best_sc > 0 else 1.0)
    labels.append({
        "branch_key": kkey,
        "read_best_score": best_sc,
        "read_second_score": second_sc,
        "read_unique_winner": int(unique),
        "read_margin_ratio": ratio,
        "read_label_winner": (best_pn if unique else ""),
    })

with open(READ_LABELS_CSV, "w", newline="") as out:
    cols = ["branch_key", "read_best_score", "read_second_score", "read_unique_winner", "read_margin_ratio", "read_label_winner"]
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in labels:
        w.writerow(r)
print("Saved:", READ_LABELS_CSV, "| rows:", len(labels))

ng_choice = {}
with open(DEC_NG_SAMPLE, "r", newline="") as f:
    rd = csv.DictReader(f)
    for r in rd:
        ng_choice[r["branch_key"]] = (r.get("RASTRO_choice") or "").strip()

eval_rows = []
comp = 0
agree = 0
for r in labels:
    key = r["branch_key"]
    if r["read_unique_winner"] != 1:
        eval_rows.append({
            "branch_key": key,
            "read_unique_winner": 0,
            "read_margin_ratio": r["read_margin_ratio"],
            "read_label_winner": r["read_label_winner"],
            "RASTRO_choice": ng_choice.get(key, ""),
            "RASTRO_agrees_label": "",
        })
        continue
    comp += 1
    ng = ng_choice.get(key, "")
    a = int(ng == r["read_label_winner"])
    agree += a
    eval_rows.append({
        "branch_key": key,
        "read_unique_winner": 1,
        "read_margin_ratio": r["read_margin_ratio"],
        "read_label_winner": r["read_label_winner"],
        "RASTRO_choice": ng,
        "RASTRO_agrees_label": a,
    })

with open(EVAL_OUT, "w", newline="") as out:
    cols = ["branch_key", "read_unique_winner", "read_margin_ratio", "read_label_winner", "RASTRO_choice", "RASTRO_agrees_label"]
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in eval_rows:
        w.writerow(r)
print("Saved:", EVAL_OUT, "| rows:", len(eval_rows))

print("\n=== Summary (NGRAM-only) ===")
print("Branches evaluated:", len(eval_keys))
unique_cnt = sum(1 for r in labels if r["read_unique_winner"] == 1)
print("Unique read-winner (no tie, score>0):", unique_cnt, "/", len(labels), f"({unique_cnt/len(labels):.3f})")
if comp:
    print("RASTRO agreement vs label:", agree, "/", comp, f"({agree/comp:.3f})")

def ratio_break(thr):
    sub = [r for r in eval_rows if r["read_unique_winner"] == 1 and float(r["read_margin_ratio"]) >= thr and r["RASTRO_agrees_label"] != ""]
    if not sub:
        return
    a = sum(int(x["RASTRO_agrees_label"]) for x in sub)
    print(f"ratio>={thr}: n={len(sub)} | RASTRO={a/len(sub):.3f}")

for thr in [1.2, 2.0, 5.0, 10.0]:
    ratio_break(thr)

# RANDOM CHOICE BASELINE (seeded)

random.seed(SEED)
OUT_EVAL_RANDOM_SEEDED = "/content/ablation_eval_random_seeded.csv"
OUT_SUMMARY_RANDOM_SEEDED = "/content/ablation_summary_random_seeded.txt"

labels_dict = {}
with open(READ_LABELS_CSV, "r", newline="") as f:
    rd = csv.DictReader(f)
    for r in rd:
        labels_dict[r["branch_key"]] = r
print("Loaded labels:", len(labels_dict))

candidates = defaultdict(list)
with open(CAND_CSV, "r", newline="") as f:
    rd = csv.DictReader(f)
    for row in rd:
        k = row["branch_key"]
        pn = (row.get("path_nodes") or "").strip()
        if pn:
            candidates[k].append(pn)
print("Loaded candidate sets:", len(candidates))

rows = []
comp = 0
agree = 0
for key, lab in labels_dict.items():
    cand_list = list(dict.fromkeys(candidates.get(key, [])))
    if len(cand_list) < 2:
        continue
    choice = random.choice(cand_list)
    unique = int(lab["read_unique_winner"])
    ratio = float(lab["read_margin_ratio"])
    label = lab["read_label_winner"]
    if unique == 1 and label:
        comp += 1
        a = int(choice == label)
        agree += a
    else:
        a = ""
    rows.append({
        "branch_key": key,
        "read_unique_winner": unique,
        "read_margin_ratio": ratio,
        "read_label_winner": label,
        "random_choice": choice,
        "random_agrees_label": a,
    })

with open(OUT_EVAL_RANDOM_SEEDED, "w", newline="") as out:
    cols = ["branch_key", "read_unique_winner", "read_margin_ratio", "read_label_winner", "random_choice", "random_agrees_label"]
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

summary_lines = []
summary_lines.append("=== Summary (RANDOM SEEDED) ===")
summary_lines.append(f"Comparable (unique label & decision exists): {comp}")
if comp:
    summary_lines.append(f"Agreement vs label: {agree} / {comp} ({agree/comp:.3f})")

def ratio_break_seeded(thr):
    sub = [r for r in rows if r["read_unique_winner"] == 1 and r["random_agrees_label"] != "" and float(r["read_margin_ratio"]) >= thr]
    if not sub:
        return
    a = sum(int(x["random_agrees_label"]) for x in sub)
    summary_lines.append(f"ratio>={thr}: n={len(sub)} | random={a/len(sub):.3f}")

for thr in [1.2, 2.0, 5.0, 10.0]:
    ratio_break_seeded(thr)

with open(OUT_SUMMARY_RANDOM_SEEDED, "w") as f:
    f.write("\n".join(summary_lines))
print("Wrote:", OUT_SUMMARY_RANDOM_SEEDED)
for line in summary_lines:
    print(line)

# RANDOM CHOICE BASELINE (unseeded multi-run)

N_RUNS = 100
OUT_EVAL_RANDOM = "/content/ablation_eval_random.csv"
OUT_SUMMARY_RANDOM = "/content/ablation_summary_random.txt"

all_agreements = []
last_rows = []
for run in range(N_RUNS):
    rows = []
    comp = 0
    agree = 0
    for key, lab in labels_dict.items():
        cand_list = list(dict.fromkeys(candidates.get(key, [])))
        if len(cand_list) < 2:
            continue
        choice = random.choice(cand_list)
        unique = int(lab["read_unique_winner"])
        ratio = float(lab["read_margin_ratio"])
        label = lab["read_label_winner"]
        if unique == 1 and label:
            comp += 1
            a = int(choice == label)
            agree += a
        else:
            a = ""
        rows.append({
            "branch_key": key,
            "read_unique_winner": unique,
            "read_margin_ratio": ratio,
            "read_label_winner": label,
            "random_choice": choice,
            "random_agrees_label": a,
        })
    if comp:
        all_agreements.append(agree / comp)
    last_rows = rows

with open(OUT_EVAL_RANDOM, "w", newline="") as out:
    cols = ["branch_key", "read_unique_winner", "read_margin_ratio", "read_label_winner", "random_choice", "random_agrees_label"]
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in last_rows:
        w.writerow(r)

summary_lines = []
summary_lines.append("=== Summary (RANDOM BASELINE) ===")
summary_lines.append(f"Number of random runs: {N_RUNS}")
if all_agreements:
    mean_agreement = statistics.mean(all_agreements)
    if len(all_agreements) > 1:
        std_agreement = statistics.stdev(all_agreements)
    else:
        std_agreement = 0.0
    summary_lines.append(f"Agreement vs label: {mean_agreement:.3f} ± {std_agreement:.3f}")
    summary_lines.append(f"Min agreement: {min(all_agreements):.3f}")
    summary_lines.append(f"Max agreement: {max(all_agreements):.3f}")

def ratio_break_unseeded(thr):
    agreements = []
    for _ in range(N_RUNS):
        comp = 0
        agree = 0
        for key, lab in labels_dict.items():
            cand_list = list(dict.fromkeys(candidates.get(key, [])))
            if len(cand_list) < 2:
                continue
            unique = int(lab["read_unique_winner"])
            ratio = float(lab["read_margin_ratio"])
            label = lab["read_label_winner"]
            if unique == 1 and label and ratio >= thr:
                choice = random.choice(cand_list)
                comp += 1
                agree += int(choice == label)
        if comp:
            agreements.append(agree / comp)
    if agreements:
        summary_lines.append(f"ratio>={thr}: {statistics.mean(agreements):.3f} ± {statistics.stdev(agreements):.3f}")

for thr in [1.2, 2.0, 5.0, 10.0]:
    ratio_break_unseeded(thr)

with open(OUT_SUMMARY_RANDOM, "w") as f:
    f.write("\n".join(summary_lines))
print("Wrote:", OUT_SUMMARY_RANDOM)
for line in summary_lines:
    print(line)

# GREEDY BASELINE

DEC_GREEDY_FULL = "/content/branch_decisions_greedy.csv"
DEC_GREEDY_SAMPLE = "/content/ablation_decisions_greedy_sample.csv"
EVAL_GREEDY_OUT = "/content/ablation_eval_greedy.csv"

groups = defaultdict(list)
with open(CAND_CSV, "r", newline="") as f:
    rd = csv.DictReader(f)
    for row in rd:
        key = row["branch_key"]
        ext_bases = int(row.get("ext_bases", "0") or 0)
        path_nodes = (row.get("path_nodes") or "").strip()
        groups[key].append({
            "branch_key": key,
            "branch_unitig": row["branch_unitig"],
            "branch_side": row["branch_side"],
            "path_nodes": path_nodes,
            "ext_bases": ext_bases,
        })

rows_out = []
for key, cand_list in groups.items():
    if not cand_list:
        continue
    cand_list_sorted = sorted(cand_list, key=lambda r: (-r["ext_bases"], r["path_nodes"]))
    best = cand_list_sorted[0]
    best_len = best["ext_bases"]
    second_len = cand_list_sorted[1]["ext_bases"] if len(cand_list_sorted) > 1 else 0
    margin = best_len - second_len
    margin_ratio = (best_len / second_len) if second_len > 0 else (float("inf") if best_len > 0 else 1.0)
    rows_out.append({
        "branch_key": key,
        "branch_unitig": best["branch_unitig"],
        "branch_side": best["branch_side"],
        "n_candidates": len(cand_list),
        "greedy_choice": best["path_nodes"],
        "greedy_len_best": best_len,
        "greedy_len_second": second_len,
        "greedy_margin": margin,
        "greedy_margin_ratio": margin_ratio,
    })

with open(DEC_GREEDY_FULL, "w", newline="") as out:
    cols = ["branch_key", "branch_unitig", "branch_side", "n_candidates", "greedy_choice", "greedy_len_best", "greedy_len_second", "greedy_margin", "greedy_margin_ratio"]
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)
print("Saved greedy decisions:", DEC_GREEDY_FULL, "| rows:", len(rows_out))

rng = random.Random(SEED)
reservoir = []
seen = 0
with open(DEC_GREEDY_FULL, "r", newline="") as f:
    rd = csv.DictReader(f)
    for row in rd:
        seen += 1
        if len(reservoir) < N_SAMPLE:
            reservoir.append(row)
        else:
            j = rng.randrange(seen)
            if j < N_SAMPLE:
                reservoir[j] = row

sample_keys = [r["branch_key"] for r in reservoir]
Path(SAMPLE_KEYS_TXT).write_text("\n".join(sample_keys) + "\n")
print(f"\nSampled {len(sample_keys)} branch-sides (from {seen}).")
print("Saved:", SAMPLE_KEYS_TXT)

with open(DEC_GREEDY_SAMPLE, "w", newline="") as out:
    cols = ["branch_key", "branch_unitig", "branch_side", "n_candidates", "greedy_choice", "greedy_len_best", "greedy_len_second", "greedy_margin", "greedy_margin_ratio"]
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in reservoir:
        w.writerow({c: r.get(c, "") for c in cols})
print("Saved:", DEC_GREEDY_SAMPLE)

greedy_choice = {}
with open(DEC_GREEDY_SAMPLE, "r", newline="") as f:
    rd = csv.DictReader(f)
    for r in rd:
        greedy_choice[r["branch_key"]] = (r.get("greedy_choice") or "").strip()

eval_rows = []
comp = 0
agree = 0
with open(READ_LABELS_CSV, "r", newline="") as f:
    rd = csv.DictReader(f)
    labels_list = list(rd)

for r in labels_list:
    key = r["branch_key"]
    unique = int(r["read_unique_winner"])
    ratio = float(r["read_margin_ratio"])
    label = (r["read_label_winner"] or "").strip()
    if unique != 1:
        eval_rows.append({
            "branch_key": key,
            "read_unique_winner": 0,
            "read_margin_ratio": ratio,
            "read_label_winner": label,
            "greedy_choice": greedy_choice.get(key, ""),
            "greedy_agrees_label": "",
        })
        continue
    comp += 1
    gc = greedy_choice.get(key, "")
    a = int(gc == label)
    agree += a
    eval_rows.append({
        "branch_key": key,
        "read_unique_winner": 1,
        "read_margin_ratio": ratio,
        "read_label_winner": label,
        "greedy_choice": gc,
        "greedy_agrees_label": a,
    })

with open(EVAL_GREEDY_OUT, "w", newline="") as out:
    cols = ["branch_key", "read_unique_winner", "read_margin_ratio", "read_label_winner", "greedy_choice", "greedy_agrees_label"]
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in eval_rows:
        w.writerow(r)
print("Saved:", EVAL_GREEDY_OUT, "| rows:", len(eval_rows))

print("\n=== Summary (GREEDY) ===")
print("Branches evaluated:", len(sample_keys))
unique_cnt = sum(1 for r in labels_list if int(r["read_unique_winner"]) == 1)
print("Unique read-winner (no tie, score>0):", unique_cnt, "/", len(labels_list), f"({unique_cnt/len(labels_list):.3f})")
if comp:
    print("greedy agreement vs label:", agree, "/", comp, f"({agree/comp:.3f})")

def ratio_break_greedy(thr):
    sub = [r for r in eval_rows if r["read_unique_winner"] == 1 and float(r["read_margin_ratio"]) >= thr and r["greedy_agrees_label"] != ""]
    if not sub:
        return
    a = sum(int(x["greedy_agrees_label"]) for x in sub)
    print(f"ratio>={thr}: n={len(sub)} | greedy={a/len(sub):.3f}")

for thr in [1.2, 2.0, 5.0, 10.0]:
    ratio_break_greedy(thr)

# Combined summaries

FILES = [
    ("NGRAM", "/content/ablation_eval_ngram.csv"),
    ("RANDOM_SEEDED", "/content/ablation_eval_random_seeded.csv"),
    ("RANDOM", "/content/ablation_eval_random.csv"),
    ("GREEDY", "/content/ablation_eval_greedy.csv"),
]
OUT_TXT = "/content/ablation_summaries.txt"
THRS = [1.2, 2.0, 5.0, 10.0]

def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except:
        return None

def get_first_existing(cols, names):
    for n in names:
        if n in cols:
            return n
    return None

def summarize(tag, path):
    if not os.path.exists(path):
        return [f"=== Summary ({tag}) ===", f"Missing file: {path}", ""]
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    if not rows:
        return [f"=== Summary ({tag}) ===", "EMPTY file.", ""]
    cols = set(rows[0].keys())
    agree_col = get_first_existing(cols, [
        "RASTRO_agrees_label",
        "random_agrees_label",
        "greedy_agrees_label",
        "ngram_agrees_label",
    ])
    scored_col = get_first_existing(cols, ["ngram_scored", "greedy_scored"])
    choice_col = get_first_existing(cols, [
        "RASTRO_choice",
        "random_choice",
        "greedy_choice",
        "choice",
    ])
    if agree_col is None:
        return [f"=== Summary ({tag}) ===", f"Could not find agreement column in {path}", f"Found columns: {list(rows[0].keys())}", ""]
    def decision_exists(r):
        if scored_col is not None:
            return str(r.get(scored_col, "0")) == "1"
        if choice_col is not None:
            return (r.get(choice_col, "") or "").strip() != ""
        return False
    comp = [
        r for r in rows
        if str(r.get("read_unique_winner", "0")) == "1"
        and decision_exists(r)
        and (r.get(agree_col, "") != "")
    ]
    comp_n = len(comp)
    agree_n = sum(1 for r in comp if str(r.get(agree_col, "0")) == "1")
    lines = []
    lines.append(f"=== Summary ({tag}) ===")
    lines.append(f"Comparable (unique label & decision exists): {comp_n}")
    if comp_n:
        lines.append(f"Agreement vs label: {agree_n} / {comp_n} ({agree_n/comp_n:.3f})")
    else:
        lines.append("Agreement vs label: 0 / 0 (0.000)")
    for thr in THRS:
        sub = []
        for r in comp:
            rr = fnum(r.get("read_margin_ratio", ""))
            if rr is not None and rr >= thr:
                sub.append(r)
        if not sub:
            continue
        a = sum(1 for r in sub if str(r.get(agree_col, "0")) == "1")
        lines.append(f"ratio>={thr}: n={len(sub)} | {tag.lower()}={(a/len(sub)):.3f}")
    lines.append("")
    return lines

all_lines = []
for tag, path in FILES:
    all_lines.extend(summarize(tag, path))

with open(OUT_TXT, "w") as f:
    f.write("\n".join(all_lines))
print("Wrote:", OUT_TXT)
print("\n".join(all_lines))