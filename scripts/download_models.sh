# Create folders
mkdir -p ./pretrained_models
mkdir -p ./pretrained_models/sadtalker
mkdir -p ./pretrained_models/liveportrait
mkdir -p ./pretrained_models/liveportrait/base_models
mkdir -p ./pretrained_models/liveportrait/retargeting_models

# Sadtalker
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors -O  ./pretrained_models/sadtalker/SadTalker_V0.0.2_256.safetensors

# LivePortrait
wget -nc https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait/base_models/appearance_feature_extractor.pth -O ./pretrained_models/liveportrait/base_models/appearance_feature_extractor.pth
wget -nc https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait/base_models/motion_extractor.pth -O ./pretrained_models/liveportrait/base_models/motion_extractor.pth
wget -nc https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait/base_models/warping_module.pth -O ./pretrained_models/liveportrait/base_models/warping_module.pth
wget -nc https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait/base_models/spade_generator.pth -O ./pretrained_models/liveportrait/base_models/spade_generator.pth
wget -nc https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.pth -O ./pretrained_models/liveportrait/retargeting_models/stitching_retargeting_module.pth

# MappingNet
wget -nc https://huggingface.co/mustafakara/liveportrait_talker/resolve/main/pretrained_models/mappingnet_basic_64inp_onlylip.pt -O ./pretrained_models/mappingnet_basic_64inp_onlylip.pt
