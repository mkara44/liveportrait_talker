from omegaconf import OmegaConf
from argparse import ArgumentParser

from src.modules.preprocess import Preprocess
from src.modules.audio2coeff import Audio2Coeff
from src.modules.map2lp import Map2LivePortrait
from src.modules.lp_render import LivePortraitRender
from src.modules.save_result import SaveResult


def main(args):
    cfg = OmegaConf.load(args.config_path)
    cfg = cfg.inference
    print("Config file is loaded succesfully!")

    preprocess = Preprocess(device=cfg.device,
                            fps=cfg.fps,
                            sadtalker_checkpoint_path=cfg.sadtalker_checkpoint_path,
                            **cfg.preprocess)
    
    batch = preprocess(source_path=args.source_path,
                       audio_path=args.audio_path)
    
    audio2coeff = Audio2Coeff(device=cfg.device,
                              sadtalker_checkpoint_path=cfg.sadtalker_checkpoint_path)
    
    batch = audio2coeff(batch=batch)
    
    map2lp = Map2LivePortrait(device=cfg.device,
                              semantic_radius=cfg.map2lp.semantic_radius,
                              mappingnet_model_path=cfg.map2lp.mappingnet_model_path,
                              mappingnet_cfg=cfg.map2lp.mappingnet_cfg)
    batch = map2lp(batch=batch)

    lp_render = LivePortraitRender(device=cfg.device,
                                   liveportrait_cfg=cfg.lp_render.liveportrait_cfg)
    batch = lp_render(batch=batch)

    save_result = SaveResult(save_path=args.save_path)
    save_result(batch=batch)

    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--source_path", type=str, help="path to source image/video")
    parser.add_argument("--audio_path", type=str, help="path to audio")
    parser.add_argument("--save_path", type=str, help="path to audio")
    args = parser.parse_args()

    main(args)