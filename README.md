# LivePortraitTalker
<p align="center">
  <img src="./assets/merged.gif" alt="showcase">
</p>
## Introduction
This is a side project combining the pretrained models of [SadTalker](https://arxiv.org/abs/2211.12194) and [LivePortrait](https://arxiv.org/abs/2407.03168).

![LivePortraitTalker Architecture](assets/arch.png)

The pretrained models in the green boxes are from Sadtalker, the red boxes are used from LivePortrait repository. The MappintNet architecture in the purple box is taken from Sadtalker and trained. The VoxCeleb2 dataset was used to train MappingNet. Due to GPU prices, the model was trained using approximately 2000 videos (<2% of the dataset). Therefore, the results may not be consistent and high quality. However, this work proves the concept. 

## Installation
- Python 3.6+
- `pip install -r requirements.txt`

## Inference

## Head Pose Generation

## Acknowledgements
