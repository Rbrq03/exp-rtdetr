
## Setup

```shell

conda create -n jnexp python=3.9
conda activate jnexp

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install natten==0.17.1+torch230cu118 -f https://shi-labs.com/natten/wheels

pip install -r requirements.txt
```

## exp

### exp1-EfficientViT-L2 NA1D Encoder

### exp2-EfficientViT-L2 NA Decoder
