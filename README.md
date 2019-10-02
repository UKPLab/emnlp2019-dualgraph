# Enhancing AMR-to-Text Generation with Dual Graph Representations
This repository contains the code for the EMNLP-IJCNLP 2019 paper: "Enhancing AMR-to-Text Generation with Dual Graph Representations".

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

This project is implemented using the framework [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and the library [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric). Please, refer to their website for further details on the installation and dependencies.

## Environments and Dependencies

- python 3
- PyTorch 1.1.0
- PyTorch Geometric 1.3.0
- nltk
- parsimonious

## Datasets

In our experiments, we use the following datasets:  [LDC2015E86](https://amr.isi.edu/download.html) and [LDC2017T10](https://amr.isi.edu/download.html).

## Preprocess

First, convert the dataset into the format required for the model.

For the LDC2015E86 dataset, run:
```
./preprocess_LDC2015E86.sh <dataset_folder> <glove_emb_file>
```
For the LDC2017T10 dataset, run:
```
./preprocess_LDC2017T10.sh <dataset_folder> <glove_emb_file>
```


## Training
For traning the model using the LDC2015E86 dataset, execute:
```
./train_LDC2015E86.sh <gpu_id> <gnn_type> <gnn_layers> <start_decay_steps> <decay_steps>
```

For the LDC2017T10 dataset, execute:
```
./train_LDC2017T10.sh <gpu_id> <gnn_type> <gnn_layers> <start_decay_steps> <decay_steps>
```
Options for `<gnn_type>` are `ggnn`, `gat` or `gin`. `<gnn_layers>` is the number of graph layers. Refer to [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) for `<start_decay_steps>` and `<decay_steps>`.

We lower the learning rate during training, after some epochs, as in Konstas et al. (2017).

Examples:
```
./train_LDC2015E86.sh 0 gin 2 6720 4200
./train_LDC2017T10.sh 0 ggnn 5 14640 10980 
```

## Trained models

- GIN-DualGraph trained on LDC2015E86 training set - BLEU on test set: 24.60 ([download](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/emnlp19_dualgraph/model_gin_ldc2015e86.pt))
- GAT-DualGraph trained on LDC2015E86 training set - BLEU on test set: 24.98 ([download](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/emnlp19_dualgraph/model_gat_ldc2015e86.pt))
- GGNN-DualGraph trained on LDC2015E86 training set - BLEU on test set: 25.01 ([download](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/emnlp19_dualgraph/model_ggnn_ldc2015e86.pt))
- GIN-DualGraph trained on LDC2017T10 training set - BLEU on test set: 28.05 ([download](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/emnlp19_dualgraph/model_gin_ldc2017t10.pt))
- GAT-DualGraph trained on LDC2017T10 training set - BLEU on test set: 27.26 ([download](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/emnlp19_dualgraph/model_gat_ldc2017t10.pt))
- GGNN-DualGraph trained on LDC2017T10 training set - BLEU on test set: 28.26 ([download](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/emnlp19_dualgraph/model_ggnn_ldc2017t10.pt))

## Decoding

For decode on the test set, run:
```
./decode.sh <gpu_id> <model> <nodes_file> <node1_file> <node2_file> <output>
```

Example:
```
./decode.sh 0 model_ggnn_ldc2015e86.pt test-amr-nodes.txt test-amr-node1.txt test-amr-node2.txt output-ggnn-test-ldc2015e86.txt
```

## More
For more details regading hyperparameters, please refer to [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

Contact person: Leonardo Ribeiro, ribeiro@aiphes.tu-darmstadt.de

## Citation

```
@inproceedings{ribeiro19-dualgraph,
    title = {Enhancing {AMR}-to-Text Generation with Dual Graph Representations},
    author = {Ribeiro, Leonardo F. R. and Gardent, Claire and Gurevych, Iryna},
    booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing ({EMNLP}}
    pages = {to appear},
    Xmonth = nov,
    year = {2019},
publisher = "Association for Computational Linguistics",
    address = {Hong Kong, China},
}
```


