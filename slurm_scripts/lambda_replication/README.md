# megaDNA LAMBDA_v1 replication

Driver scripts that reproduce the LAMBDA paper surfaces for megaDNA, mirroring
the ProkBERT / DNABERT-2 / NT-v2 `lambda_replication/` pipelines. Two commands
run the whole thing: one submits all per-seed jobs, the other picks the best
seed per variant and submits every inference job.

**megaDNA is a GENERATIVE byte-level model and does NOT finetune.**
Classification = extract embeddings from the frozen pretrained backbone, then
train a linear probe + a 3-layer NN on top. So the trainable / per-seed stage
here IS the embedding analysis (`embedding_analysis_megadna.py`), and the
"winner" is the best seed's trained 3-layer NN (its `three_layer_nn_pretrained.pt`
checkpoint + `..._scaler.pkl`). megaDNA's native context is ~96 kb, so all
LAMBDA windows (2k / 4k / 8k) fit comfortably.

## Layout

```
megaDNA/                              # repo root
  embedding_analysis_megadna.py      # per-seed train stage (frozen backbone -> linear probe + 3-layer NN)
  inference_megadna.py               # prediction entry point (backbone + NN classifier + scaler)
  slurm_scripts/
    lambda_replication/              # <-- this directory
      lambda_replication.conf        # all paths + hyperparameters (edit this)
      run_lambda_training.sh         # STAGE 1 launcher (per-seed embedding analysis)
      run_lambda_inference.sh        # STAGE 2 launcher (winners + inference)
      lambda_finetune_job.sh         # sbatch body: one (variant, seed) embedding analysis
      lambda_inference_job.sh        # sbatch body: one prediction surface
      select_best_model.py           # best-of-N seed by test NN MCC -> winners.json
      print_winner_exports.py        # winners.json -> shell exports (used by inference job)
      check_training.sh              # Stage 1 completeness report
      check_inference.sh             # Stage 2 completeness report
```

There is intentionally **no** separate embedding job in Stage 2 (the embedding
analysis is Stage 1), and **no** local genome-wide clustering / aggregation job:
genome-level analysis is done CENTRALLY by the harvest pipeline. This repo's
`genome_predictions_megadna.py` clustering path is NOT used.

## Outputs

Everything lands under `OUTPUT_DIR` from the config, in a single per-length tree:

```
/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/megaDNA/outputs/
  <ws>/                                  # 2k, 4k, 8k
    finetune/<variant>/seed-<N>/         # embedding_analysis_results.json
                                         #   three_layer_nn_pretrained.pt
                                         #   three_layer_nn_pretrained_scaler.pkl
    winners.json                         # best seed per variant (by pretrained_nn_mcc)
    inference/<variant>/                 # test_predictions.csv, fpr_predictions.csv,
                                         #   gc_control_predictions.csv, fnr_predictions.csv,
                                         #   genome_wide_<stem>_predictions.csv (+ _metrics.json)
  logs/                                  # all SLURM stdout/stderr
```

The `<diag>_predictions.csv` / `genome_wide_<stem>_predictions.csv` names and the
sibling `*_metrics.json` files are the CANONICAL names the central harvest
aggregator globs for (`genome_wide_` is the genome-wide glob key).

## Full sweep

```bash
cd /path/to/megaDNA

# 1. edit slurm_scripts/lambda_replication/lambda_replication.conf if needed
#    (confirm LAMBDA_BASE, OUTPUT_DIR, MODEL_PATH).
# 2. submit all per-seed jobs (variant x seed x {2k,4k,8k})
bash slurm_scripts/lambda_replication/run_lambda_training.sh

# 3. wait until every job is done
squeue -u $USER
bash slurm_scripts/lambda_replication/check_training.sh

# 4. pick winners + submit all inference jobs
bash slurm_scripts/lambda_replication/run_lambda_inference.sh

# 5. wait; then verify
bash slurm_scripts/lambda_replication/check_inference.sh
```

## Smoke test (run this ONE thing first)

Run a single seed for the single variant at 2k, then confirm the output
structure. From the repo root on Biowulf:

```bash
cd /path/to/megaDNA
REPO_ROOT="$(pwd)"
LAMBDA_DIR="/vf/users/Irp-jiang/share/lindseylm/LAMBDA_v1/train_val_test/2k"
OUT="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/megaDNA/outputs/2k"
MODEL_PATH="/data/lindseylm/GLM_EVALUATIONS/MODELS/MEGADNA/megaDNA/megaDNA_phage_145M.pt"
mkdir -p "${OUT}/../logs"

sbatch --job-name=smoke_ft_2k_megadna_s1 \
  --partition=gpu --gres=gpu:a100:1 --mem=64g --time=8:00:00 --cpus-per-task=8 \
  --output="${OUT}/../logs/smoke_%j.out" --error="${OUT}/../logs/smoke_%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=megadna,REPL_OUTPUT_DIR=${OUT},LAMBDA_DIR=${LAMBDA_DIR},VARIANT=megadna,SEED=1,LEN=2k,MAX_LENGTH=2200,MODEL_PATH=${MODEL_PATH},LAYER=middle,POOLING=mean,NN_EPOCHS=100,NN_HIDDEN_DIM=256,NN_LR=0.001,BATCH_SIZE=8" \
  slurm_scripts/lambda_replication/lambda_finetune_job.sh
```

When it finishes, expect:

```
outputs/2k/finetune/megadna/seed-1/embedding_analysis_results.json   # has pretrained_nn_mcc
outputs/2k/finetune/megadna/seed-1/three_layer_nn_pretrained.pt
outputs/2k/finetune/megadna/seed-1/three_layer_nn_pretrained_scaler.pkl
```

## Notes / assumptions

- **No model/experiment code is modified.** The jobs call the existing
  `embedding_analysis_megadna.py` / `inference_megadna.py` entry points with
  explicit output paths/names. Both already read `dev.csv` OR `val.csv`, so no
  data staging is needed.
- **`MODEL_PATH`** defaults to the 145M phage checkpoint path from the existing
  megaDNA multiseed wrapper. Confirm/override in the config.
- **`SEEDS`** defaults to `1 2 3 4 5` (matches the other LAMBDA model repos). The
  standalone megaDNA multiseed wrapper used `1..10`; adjustable in the config.
- **`LAYER`** defaults to `middle` (the multiseed wrapper's convention; 256-d
  embeddings). `local`/`global`/`all` are also valid.
- **`MAX_LENGTH_<LEN>`** (config) are per-window base-pair truncation caps set
  just above each window (2200 / 4200 / 8200). megaDNA's standalone scripts
  default to 96000; the smaller caps are equivalent for these windows (no LAMBDA
  sequence exceeds them) and avoid reserving the full 96 kb budget. Set all three
  to 96000 to match the standalone scripts exactly.
- **Environment:** conda env `megadna`, `module load conda` + `module load
  CUDA/12.8`, bare `source activate`. Following the existing megaDNA scripts, the
  jobs do NOT set `PYTHONNOUSERSITE` and do NOT use HF offline vars (the backbone
  is a local `.pt`, not a HuggingFace download).
```
