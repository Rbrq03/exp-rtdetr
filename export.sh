python tools/export_onnx.py -c /opt/data/private/hjn/fna-detection/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdenatr/rtdetrv2_evitl2_coco.yaml -r /opt/data/private/hjn/fna-detection/RT-DETR/output/rtdetr-efficientViT-l2/best.pth --check --simplify
trtexec --onnx=./model.onnx --workspace=4096 --saveEngine=model.engine --avgRuns=100 --fp16
# mv ./model.onnx /opt/data/private/hjn/fna-detection/RT-DETR/output/rtdetr-efficientViT-l2/
# mv ./model.engine /opt/data/private/hjn/fna-detection/RT-DETR/output/rtdetr-efficientViT-l2/