CUDA_VISIBLE_DEVICES=1 python moe_v1.py --config ./configs/cora/engine.yaml --early   >>output_1.log
CUDA_VISIBLE_DEVICES=1 python moe_v1.py --config ./configs/citeseer/engine.yaml --early  >>output_1.log
CUDA_VISIBLE_DEVICES=1 python moe_v1.py --config ./configs/wikics/engine.yaml --early  >>output_1.log
CUDA_VISIBLE_DEVICES=1 python moe_v1.py --config ./configs/arxiv/engine.yaml --early   >>output_1.log
CUDA_VISIBLE_DEVICES=1 python moe_v1.py --config ./configs/arxiv_2023/engine.yaml --early  >>output_1.log
CUDA_VISIBLE_DEVICES=1 python moe_v1.py --config ./configs/products/engine.yaml --early  >>output_1.log
CUDA_VISIBLE_DEVICES=1 python moe_v1.py --config ./configs/photo/engine.yaml --early  >>output_1.log
