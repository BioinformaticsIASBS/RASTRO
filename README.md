## Overview

RASTRO is a reference-free framework for making local branch-selection
decisions in unitig graphs. It evaluates competing candidate extensions at
branching points using sequence-level information rather than relying only on
graph topology.

The framework contains two complementary approaches:

- **RASTRO-S** uses a sample-specific n-gram statistical model learned from the
  unitigs of the target dataset.
- **RASTRO-L** uses the pre-trained Evo2 genomic language model to evaluate
  candidate extensions based on learned genomic sequence patterns.

For a branching point with multiple candidate continuations, each method
assigns a score to the candidate extensions. The candidate with the lowest
bits-per-base (bpb) score is selected.

## RASTRO-S

RASTRO-S is a lightweight, sample-specific statistical model designed to score
candidate paths at branching points in a unitig graph.

The model is trained on all unitigs contained in the corresponding Logan FASTA
file, allowing it to learn local sequence patterns specific to the target
dataset. For each candidate extension at a branching point, RASTRO-S evaluates
how statistically consistent the extension is with the sequence patterns
learned from the dataset.

RASTRO-S uses an n-th order Markov model, where the probability of the next DNA
base depends on the preceding n bases. Candidate extensions are scored using
bits per base (bpb), derived from the model's next-base probabilities. Lower
bpb values indicate that the candidate extension is more consistent with the
sequence patterns learned from the sample.

The model considers only the standard DNA alphabet (A, C, G, T). Characters
outside this alphabet are excluded from training and scoring.

## RASTRO-L

RASTRO-L uses the pre-trained Evo2 genomic language model to score candidate
paths at branching points in a unitig graph.

Evo2 is an autoregressive genomic language model trained using next-token
prediction. In this setting, the model predicts the next DNA base based on the
sequence observed so far. RASTRO-L uses these next-base predictions to
evaluate the plausibility of each candidate extension.

Each candidate path is scored by evaluating the probability of its sequence
given the preceding sequence context. The resulting score is normalized by
the extension length and expressed as bits per base (bpb), allowing candidate
paths of different lengths to be compared. Lower bpb values indicate greater
sequence plausibility according to Evo2.

At each branching point, the candidate with the lowest bpb score is selected.

## Read-Based Evaluation

RASTRO decisions are evaluated independently of the final assembled contigs.
The original sequencing reads are used as the source of evidence for
determining which candidate extension is supported by the sample.

The evaluation procedure consists of the following steps:

1. Load the Logan unitig graph and construct the traversal structures required
   for candidate-path exploration.
2. Identify branching points and generate the valid candidate extensions.
3. Select the candidate with the lowest bpb score when at least two comparable
   candidates are available.
4. For each candidate, construct the sequence spanning the junction between
   the existing graph context and the candidate extension.
5. Extract boundary k-mers from this junction and count their occurrences in
   the raw sequencing reads.
6. Determine the read-supported candidate from the boundary k-mer support.
7. Compare the RASTRO decision with the read-supported choice.

The evaluation reports both the percentage of branches for which a valid
decision can be made and the percentage of decisions that agree with the
read-supported choice. Different confidence thresholds are used to examine
how the strength of read evidence affects the results.

## Why Use Raw Reads for Evaluation?

Raw sequencing reads provide direct evidence from the sequenced sample and are
independent of the assembly decisions being evaluated.

In contrast, assembled contigs are themselves produced through graph
traversal and branch-selection decisions. Using contigs as the evaluation
reference could therefore introduce assembly-dependent bias or obscure the
original ambiguity at a branching point.

RASTRO therefore uses raw sequencing reads and boundary k-mers to evaluate
branch-selection decisions independently of the final assembly.

## Datasets

The experiments use unitig graphs and sequencing reads from two viral
datasets:

- **HIV-1:** SRR7693968
- **SARS-CoV-2:** SRR10605873

The unitig graphs are obtained from Logan. Raw sequencing reads are used for
the read-based evaluation of branch-selection decisions.
