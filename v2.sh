CUDA_VISIBLE_DEVICES=0 python moe_v2.py --config ./configs/cora/engine.yaml --early    >>output_2.log
CUDA_VISIBLE_DEVICES=0 python moe_v2.py --config ./configs/citeseer/engine.yaml --early >>output_2.log
CUDA_VISIBLE_DEVICES=0 python moe_v2.py --config ./configs/wikics/engine.yaml --early >>output_2.log
CUDA_VISIBLE_DEVICES=0 python moe_v2.py --config ./configs/arxiv/engine.yaml --early >>output_2.log
CUDA_VISIBLE_DEVICES=0 python moe_v2.py --config ./configs/arxiv_2023/engine.yaml --early >>output_2.log
CUDA_VISIBLE_DEVICES=0 python moe_v2.py --config ./configs/products/engine.yaml --early >>output_2.log
CUDA_VISIBLE_DEVICES=0 python moe_v2.py --config ./configs/photo/engine.yaml --early >>output_2.log
