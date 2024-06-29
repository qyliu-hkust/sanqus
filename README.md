# sanqus
Implementation of staleness+quantization for efficient distributed GNN training based on [SANCUS](https://github.com/chenzhao/light-dist-gnn).

#### Getting started

1. Setup a clean environment.
```
conda create --name gnn
conda activate gnn
```

2. Install dependencies.
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c dglteam dgl-cuda11.3
conda install pyg -c pyg -c conda-forge
pip install ogb
```

3. Compile and install spmm. (Optional. CUDA dev environment needed.)
```
cd spmm_cpp
python setup.py install
```

4. Prepare dataset. 
```
python prepare_data.py --dataset dataset_name --partition part_num
```
Optional dataset_name: reddit (default), ogbn-products, ogbn-arxiv, amazon-products, flickr

Graph partition number can be either 4 (default) or 8 based on your available GPUs. 

5. Train gnn model. 
```
python main.py --model model_name --dataset dataset_name --epoch 20
```
Optional model choice: gcn (default), qgcn, cgcn, gat


#### About dataset preparation
There is a known issue leading to a download error when you prepare the amazon dataset for a older `pyg` version. 

We fix the bug in file `/fix_patch/amazon_products.py`. 

You can replace yours in `PATH_TO_PYG_LIB/torch_geometric/datasets/` to avoid data downloading error. 

