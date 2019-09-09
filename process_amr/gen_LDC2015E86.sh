#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p ${ROOT_DIR}/data
REPO_DIR=${ROOT_DIR}/data/

DATA_DIR=${1}
mkdir -p ${REPO_DIR}/tmp_amr
PREPROC_DIR=${REPO_DIR}/tmp_amr
ORIG_AMR_DIR=${DATA_DIR}/data/alignments/split
mkdir -p ${REPO_DIR}/amr_ldc2015e86
FINAL_AMR_DIR=${REPO_DIR}/amr_ldc2015e86

mkdir -p ${PREPROC_DIR}/train
mkdir -p ${PREPROC_DIR}/dev
mkdir -p ${PREPROC_DIR}/test

mkdir -p ${FINAL_AMR_DIR}/train
mkdir -p ${FINAL_AMR_DIR}/dev
mkdir -p ${FINAL_AMR_DIR}/test

cat ${ORIG_AMR_DIR}/training/deft-* > ${PREPROC_DIR}/train/raw_amrs.txt
cat ${ORIG_AMR_DIR}/dev/deft-* > ${PREPROC_DIR}/dev/raw_amrs.txt
cat ${ORIG_AMR_DIR}/test/deft-* > ${PREPROC_DIR}/test/raw_amrs.txt

for SPLIT in train dev test; do
    echo "processing $SPLIT..."
    python ${ROOT_DIR}/split_amr.py ${PREPROC_DIR}/${SPLIT}/raw_amrs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${PREPROC_DIR}/${SPLIT}/graphs.txt
    python ${ROOT_DIR}/preproc_amr.py ${PREPROC_DIR}/${SPLIT}/graphs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${FINAL_AMR_DIR}/${SPLIT}/nodes.pp.txt ${FINAL_AMR_DIR}/${SPLIT}/surface.pp.txt --mode LINE_GRAPH --triples-output ${FINAL_AMR_DIR}/${SPLIT}/triples.pp.txt
    echo "done."
done
