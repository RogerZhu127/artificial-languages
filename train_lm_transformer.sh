#!/bin/bash
#SBATCH --job-name=lang_batch
#SBATCH --output=logs/%x_%j.out
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=u22
#SBATCH --qos=m5
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=y485zhu@uwaterloo.ca

GRAMMAR=$1
SPLIT=$2

mkdir -p "data-bin/${GRAMMAR}/${SPLIT}-dataset"
mkdir -p "checkpoints/${GRAMMAR}/${SPLIT}-transformer"
mkdir -p "trans-results"
mkdir -p "trans_sentence_scores"

fairseq-preprocess --only-source --trainpref "data_gen/permuted_splits/${GRAMMAR}/${SPLIT}.trn" --validpref "data_gen/permuted_splits/${GRAMMAR}/${SPLIT}.dev" --testpref "data_gen/permuted_splits/${GRAMMAR}/${SPLIT}.tst" --destdir "data-bin/${GRAMMAR}/${SPLIT}-dataset" --workers 20

fairseq-train --task language_modeling "data-bin/${GRAMMAR}/${SPLIT}-dataset" \
    --save-dir "checkpoints/${GRAMMAR}/${SPLIT}-transformer" \
    --arch transformer_lm \
    --decoder-layers 2 \
    --decoder-attention-heads 2 \
    --decoder-embed-dim 128 \
    --decoder-ffn-embed-dim 512 \
    --share-decoder-input-output-embed \
    --dropout 0.3 \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --weight-decay 0.01 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --clip-norm 0.0 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 \
    --sample-break-mode none \
    --max-tokens 4096 \
    --update-freq 16 \
    --patience 5 \
    --max-update 10000 \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --max-epoch 10

fairseq-eval-lm "data-bin/${GRAMMAR}/${SPLIT}-dataset" --path "checkpoints/${GRAMMAR}/${SPLIT}-transformer/checkpoint_best.pt" --tokens-per-sample 512 --gen-subset "valid" --output-word-probs --quiet 2> "trans-results/${GRAMMAR}.${SPLIT}.dev.txt"

fairseq-eval-lm "data-bin/${GRAMMAR}/${SPLIT}-dataset" --path "checkpoints/${GRAMMAR}/${SPLIT}-transformer/checkpoint_best.pt" --tokens-per-sample 512 --gen-subset "test" --output-word-probs --quiet 2> "trans-results/${GRAMMAR}.${SPLIT}.test.txt"

python get_sentence_scores.py -i "trans-results/${GRAMMAR}.${SPLIT}.test.txt" -O "trans_sentence_scores/"
