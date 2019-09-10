#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "./preprocess_LDC2015E86.sh <dataset_folder> <embeddings_file>"
  exit 2
fi

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

bash ${ROOT_DIR}/process_amr/gen_LDC2015E86.sh ${1}

python ${ROOT_DIR}/process_amr/generate_input_opennmt.py -i ${ROOT_DIR}/process_amr/data/amr_ldc2015e86/

mkdir -p ${ROOT_DIR}/data/ldc2015e86
mv ${ROOT_DIR}/process_amr/data/amr_ldc2015e86/dev-* ${ROOT_DIR}/data/ldc2015e86
mv ${ROOT_DIR}/process_amr/data/amr_ldc2015e86/test-* ${ROOT_DIR}/data/ldc2015e86
mv ${ROOT_DIR}/process_amr/data/amr_ldc2015e86/train-* ${ROOT_DIR}/data/ldc2015e86

rm -rf data/ldc2015e86.*

python preprocess.py -train_src data/ldc2015e86/train-amr-nodes.txt \
-train_node1 data/ldc2015e86/train-amr-node1.txt \
-train_node2 data/ldc2015e86/train-amr-node2.txt \
-train_tgt data/ldc2015e86/train-amr-tgt.txt \
-valid_src data/ldc2015e86/dev-amr-nodes.txt \
-valid_node1 data/ldc2015e86/dev-amr-node1.txt \
-valid_node2 data/ldc2015e86/dev-amr-node2.txt \
-valid_tgt data/ldc2015e86/dev-amr-tgt.txt \
-save_data data/ldc2015e86 \
-data_type graph \
-src_vocab_size 20000 \
-tgt_vocab_size 20000 \
-dynamic_dict \
-src_seq_length \
700 \
-tgt_seq_length \
700

export PYTHONPATH=${ROOT_DIR}

python tools/embeddings_to_torch.py -emb_file_enc ${2} -emb_file_dec ${2} \
-dict_file data/ldc2015e86.vocab.pt \
-output_file data/ldc2015e86.embeddings -type GloVe
