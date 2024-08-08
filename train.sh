export CUDA_DEVICE_ORDER="PCI_BUS_ID"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c /opt/data/private/hjn/fna-detection/exp/configs/rtdetrv2/rtdenatr/rtdetrv2_evitl2_coco_yolo.yaml --use-amp --seed=0 > log_rtdenatr_l2.log