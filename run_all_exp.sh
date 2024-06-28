#!/bin/bash

datasets=('reddit' 'ogbn-products' 'ogbn-arxiv' 'amazon-products')
epochs=('10' '20' '50' '100')

echo "GCN"

for dataset in "${datasets[@]}"
do
  for epoch in "${epochs[@]}"
  do
    echo "======================================================="
    echo "dataset: $dataset"
    echo "epoch: $epoch"
    python main.py --model gcn --dataset "$dataset" --epoch "$epoch"  > "cpu28/log_gcn_${dataset}_${epoch}" 2>&1
  done
done


echo "======================================================="
echo "QGCN"

for dataset in "${datasets[@]}"
do
  for epoch in "${epochs[@]}"
  do
    echo "======================================================="
    echo "dataset: $dataset"
    echo "epoch: $epoch"
    python main.py --model qgcn --dataset "$dataset" --epoch "$epoch"  > "cpu28/log_qgcn_${dataset}_${epoch}" 2>&1
  done
done
