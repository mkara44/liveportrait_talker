from omegaconf import OmegaConf
from argparse import ArgumentParser

from src.modules.preprocess import Preprocess


def main(args):
    cfg = OmegaConf.load(args.config_path)
    cfg = cfg.inference
    print("Config file is loaded succesfully!")

    preprocess = Preprocess(device=cfg.device,
                            fps=cfg.fps,
                            sadtalker_checkpoint_path=cfg.sadtalker_checkpoint_path,
                            **cfg.preprocess)
    
    data = preprocess(source_path=args.source_path,
                      audio_path=args.audio_path)
    
    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--source_path", type=str, help="path to source image/video")
    parser.add_argument("--audio_path", type=str, help="path to audio")
    args = parser.parse_args()

    main(args)