CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdenatr/rtdetrv2_evitl2_coco_yolo_official.yaml --use-amp --seed=0
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=9909 --nproc_per_node=8 tools/train.py -c /opt/data/private/hjn/fna-detection/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml --use-amp --seed=0