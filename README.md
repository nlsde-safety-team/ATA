# Generating Transferable Adversarial Examples against Vision Transformers

## Important Requirements

```
timm==0.4.12
torch
```
`Transformer-Explainability` is from `https://github.com/hila-chefer/Transformer-Explainability`, which has no relationship with any authors. We upload the necessary codes to ensure our method could run.

## Training

###### Coarse-grained Attentional Region Location

```shell
python attention_region.py --datapath=[path to imagenet validation dataset] --modelname=[training model]
```

###### Fine-grained Embedding Pixel Perturbation

```shell
python embed_position.py --datapath=[path to imagenet validation dataset] --modelname=[training model]
```

###### Train patch

```shell
python patch_train.py --datapath=[path to imagenet validation dataset] --modelname=[training model] --attnpath=[path to attention data] --embedpath=[path to embeding data]
```

## Testing

```shell
python patch_test.py --datapath=[path to imagenet validation dataset] --modelname=[testing model] --patchlog=[path to trained patch log dir]
```