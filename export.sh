#EfficientViT-L2-V
export NAME="rtdenatrx_6x"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=9909 --nproc_per_node=8 tools/train.py -c /opt/data/private/hjn/fna-detection/exp/configs/rtdetrv2/rtdenatr/${NAME}.yaml --use-amp --seed=0
python tools/export_onnx.py -c /opt/data/private/hjn/fna-detection/exp/configs/rtdetrv2/rtdenatr/${NAME}.yaml -r /opt/data/private/hjn/fna-detection/output/test/best.pth --check --simplify
trtexec --onnx=./model.onnx --workspace=4096 --saveEngine=model.engine --fp16 --iterations=3000
mv model.engine engine/${NAME}.engine
rm -rf model.onnx