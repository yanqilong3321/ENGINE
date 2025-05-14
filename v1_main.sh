CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/cora/engine.yaml --early   >>output_0.log
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/citeseer/engine.yaml --early  >>output_0.log
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/wikics/engine.yaml --early   >>output_0.log
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/arxiv/engine.yaml --early   >>output_0.log
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/arxiv_2023/engine.yaml --early >>output_0.log
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/products/engine.yaml --early  >>output_0.log
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/photo/engine.yaml --early  >>output_0.log
