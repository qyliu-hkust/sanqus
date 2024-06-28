# Copyright 2021, Zhao CHEN
# All rights reserved.
import coo_graph
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="reddit", 
                        choices=["reddit", "ogbn-products", "ogbn-arxiv", "amazon-products", "flickr"])
    parser.add_argument("--partition", type=int, default=4, choices=[4, 8])
    args = parser.parse_args()

    r = coo_graph.COO_Graph(args.dataset, full_graph_cache_enabled=True)
    r.partition(args.partition)

