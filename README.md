
## Setup

```shell

conda create -n jnexp python=3.9
conda activate jnexp

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install natten==0.17.1+torch230cu118 -f https://shi-labs.com/natten/wheels

pip install -r requirements.txt
```

## exp

pls modify the dataset path in `configs/dataset/coco_detection.yml`

### EXP-OfficialLRBS-RTDETR-EfficientViT-L2

```
bash train.sh
```
