import torch
from torch import nn
import safetensors

from src.utils.helper import load_x_from_safetensor
from src.utils.audio2coeff.cvae import CVAE
from src.utils.audio2coeff.audio_encoder import AudioEncoder


class Audio2Pose(nn.Module):
    def __init__(self, device, sadtalker_checkpoint_path, cvae_cfg, wav2lip_checkpoint=None):
        super().__init__()
        self.device = device
        self.seq_len = cvae_cfg.seq_len
        self.latent_dim = cvae_cfg.latent_size

        self.audio_encoder = AudioEncoder(wav2lip_checkpoint, device)
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.netG = CVAE(cvae_cfg) 
        self.load_netG(sadtalker_checkpoint_path)

    def load_netG(self, sadtalker_checkpoint_path):
        checkpoints = safetensors.torch.load_file(sadtalker_checkpoint_path)
        self.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2pose'), strict=False)       
        
    def __call__(self, x):
        batch = {}
        ref = x['source_coeff']                            #bs 1 70
        batch['ref'] = x['source_coeff'][:,0,-6:]  
        batch['class'] = torch.LongTensor([0]).to(self.device) #x['class']  
        bs = ref.shape[0]
        
        indiv_mels= x['indiv_mels']               # bs T 1 80 16
        indiv_mels_use = indiv_mels[:, 1:]        # we regard the ref as the first frame
        num_frames = x['num_frames']
        num_frames = int(num_frames) - 1

        #  
        div = num_frames//self.seq_len
        re = num_frames%self.seq_len
        pose_motion_pred_list = [torch.zeros(batch['ref'].unsqueeze(1).shape,
                                             dtype=batch['ref'].dtype, 
                                             device=batch['ref'].device)]

        for i in range(div):
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) #bs seq_len 512
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'])  #list of bs seq_len 6
        
        if re != 0:
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, -1*self.seq_len:,:,:,:]) #bs seq_len  512
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len-audio_emb.shape[1]
                pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1) 
                audio_emb = torch.cat([pad_audio_emb, audio_emb], 1) 
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'][:,-1*re:,:])   
        
        pose_motion_pred = torch.cat(pose_motion_pred_list, dim = 1)
        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # bs T 6

        return pose_pred