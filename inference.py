import datetime
from omegaconf import OmegaConf
from argparse import ArgumentParser

from src.modules.preprocess import Preprocess
from src.modules.audio2coeff import SadtalkerAudio2Coeff
from src.modules.map2lp import Map2LivePortrait
from src.modules.lp_render import LivePortraitRender
from src.modules.file_operations import FileOperations


def main(args):
    cfg = OmegaConf.load(args.config_path)
    cfg = cfg.inference
    print("Config File is loaded succesfully!")

    file_operations = FileOperations(device=cfg.device,
                                     save_path=args.save_path,
                                     source_path=args.source_path,
                                     audio_path=args.audio_path)
    
    preprocess = Preprocess(device=cfg.device,
                            fps=cfg.fps,
                            sadtalker_checkpoint_path=cfg.sadtalker_checkpoint_path,
                            preprocessed_inputs_exist=file_operations.preprocessed_inputs_exist,
                            **cfg.preprocess)
    
    audio2coeff = SadtalkerAudio2Coeff(device=cfg.device,
                                       sadtalker_checkpoint_path=cfg.sadtalker_checkpoint_path,
                                       **cfg.audio2coeff)
    
    map2lp = Map2LivePortrait(device=cfg.device,
                              **cfg.map2lp)
    
    lp_render = LivePortraitRender(device=cfg.device,
                                   liveportrait_cfg=cfg.lp_render.liveportrait_cfg)
    
    print("Pipeline Objects are initialized!")
    
    batch = {"source_path": args.source_path,
             "audio_path": args.audio_path,
             "head_pose_weight": args.head_pose_weight,
             "time": datetime.datetime.now().strftime("%m%d%Y-%H%M%S")}
    
    if file_operations.preprocessed_inputs_exist:
        print("Using preprocessed inputs for this source input!")
        batch = {**batch, **file_operations.load_inputs()}  
    
    pipeline = [preprocess, audio2coeff, map2lp, lp_render]
    for pipe_func in pipeline:
        batch = pipe_func(batch)

    file_operations.save(batch=batch)
    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--source_path", type=str, help="path to source image/video")
    parser.add_argument("--audio_path", type=str, help="path to audio")
    parser.add_argument("--save_path", type=str, default="./outputs", help="path to save output video")
    parser.add_argument("--head_pose_weight", type=float, default=1., help="weight to apply head pose")
    args = parser.parse_args()

    main(args)