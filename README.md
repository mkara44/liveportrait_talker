# LivePortraitTalker
LivePortraitTalker is a zero-shot talking head generation approach. It combines the pretrained models of [SadTalker](https://arxiv.org/abs/2211.12194) and [LivePortrait](https://arxiv.org/abs/2407.03168). Novelty of this repo;
- Training the mapping network of Sadtalker for LivePortrait rendering networks.
- This repo propose synthetic head pose generation which uses the inital head pose's and mappingnet outputs. 

This is just Proof of Concept of the approach, the model is only trained on 8% of the VoxCeleb2 dataset.

<p align="center">
  <img src="./assets/merged.gif" alt="showcase">
  <br /><i>Outputs of LivePortraitTalker</i>
</p>

## Introduction

<p align="center">
  <img src="./assets/arch.png" alt="LivePortraitTalker Architecture">
    <br /><i>Model Diagram</i>
</p>

The pretrained models in the green boxes are from Sadtalker, the red boxes are from LivePortrait repository. The MappintNet architecture in the purple box is taken from Sadtalker and trained. The VoxCeleb2 dataset was used to train MappingNet. Due to GPU prices, the model was trained using approximately 9000 videos (<8% of the dataset). Therefore, the results may not be consistent and high quality. However, this work proves the concept. 

## Installation
- Python 3.10+
- Install PyTorch 2.3.0, you should install compatible version with your system requirements. You can find PyTorch 2.3.0 versions [here](https://pytorch.org/get-started/previous-versions/#v230)
- `pip install -r requirements.txt`
- [FFmpeg](https://www.ffmpeg.org/) need to be installed
- [Sadtalker](https://github.com/OpenTalker/SadTalker/tree/main) and [LivePortrait](https://github.com/KwaiVGI/LivePortrait/tree/main) pretrained models must be downloaded from their repository. MappingNet can be downloaded from [here](https://huggingface.co/mustafakara/liveportrait_talker/tree/main/pretrained_models) or you can run following command to install pretrained models automatically:

```bash
sh scripts/download_models.sh
```

## Inference
There are couple of options to generate talking head; _synthetic head pose generation_, _reference head pose_, _still_, _video2video_, _pupil control_.

Don't forget to change device type from the `config.yaml` file. You need to set the `inference.device` to specify the location where the model will run: use `cuda` for GPU, `cpu` for CPU, and `mps` for MacBook Silicon.

MacBook Silicon users has to add `PYTORCH_ENABLE_MPS_FALLBACK=1` command before the python command
```bash 
PYTORCH_ENABLE_MPS_FALLBACK=1 python inference.py ...
```

### Synthetic Head Pose Generation

Most talking head papers, such as SadTalker, generate head poses from the input audio. However, I do not think that head poses have a common features with audio. Therefore, I proposed Synthetic Head Pose Generation without using audio. This approach can generate head poses more naturally then previous approaches. I will give more information about Synthetic Head Pose Generation in next sections.

```bash 
python inference.py --config_path config.yaml --source_path <path/to/source/image> --audio_path <path/to/audio> --save_path <path/to/save/folder>
```

<div align="center">
  <video src="https://github.com/user-attachments/assets/63496204-3d2c-47d1-aec5-f6b2425b602f" type="video/mp4"> </video> 
</div>

### Reference Head Pose

This option takes reference video as a input and generates talking head using poses of the person from the reference video. Once reference video is processed, head poses are saved to be used for next generation to increase inference speed. In some cases input audio and the reference head poses can be irrelevant, therefore should be used with more stable reference head poses.

```bash
python inference.py --config_path config.yaml --source_path <path/to/source/image> --audio_path <path/to/audio> --save_path <path/to/save/folder> --ref_head_pose_path <path/to/reference/video>
```

This pipeline select the initial head pose frame randomly, `ref_frames_from_zero` can be added to set the initial frame to 0;

```bash 
python inference.py --config_path config.yaml --source_path <path/to/source/image> --audio_path <path/to/audio> --save_path <path/to/save/folder> --ref_head_pose_path <path/to/reference/video> --ref_frames_from_zero
```

<div align="center">
  <video src="https://github.com/user-attachments/assets/b771aee7-96f7-4e93-ae30-114e872d0519" type="video/mp4"> </video> 
</div>

### Still

There is no head movements in this option. Only lips and blinks are generated.

```bash
python inference.py --config_path config.yaml --source_path <path/to/source/image> --audio_path <path/to/audio> --save_path <path/to/save/folder> --still
```

<div align="center">
  <video src="https://github.com/user-attachments/assets/1e0a743c-c608-4217-94c4-6736badee171" type="video/mp4"> </video> 
</div>

### Video2Video

If the video ise given as a `source_path`. The repository generates the lips using audio, while providing the head poses as the original frame.

```bash
python inference.py --config_path config.yaml --source_path <path/to/source/video> --audio_path <path/to/audio> --save_path <path/to/save/folder>
```

<div align="center">
  <video src="https://github.com/user-attachments/assets/11ec02ed-cee0-4e83-bd4f-8356cdd37035" type="video/mp4"> </video> 
</div>

### Pupil Control

Unlike Sadtalker, this repository predicts only lip expressions. Therefore, other facial expression are taken from the source image. This can be problematic if the eyes in the source image are not looking directly at the camera. Thanks to the [ComfyUI-AdvancedLivePortrait](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait), pupils can be aranged. 

```bash
python inference.py --config_path config.yaml --source_path <path/to/source/image> --audio_path <path/to/audio> --save_path <path/to/save/folder> --pupil_x <pupil/x/number> --pupil_y <pupil/y/number>
```

## Head Pose Generation
will be updated

## Acknowledgements
- [SadTalker](https://github.com/OpenTalker/SadTalker/tree/main)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait/tree/main)
- [One-Shot Free-View Neural Talking Head Synthesis](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis)
