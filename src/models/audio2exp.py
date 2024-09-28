import torch
from torch import nn
import safetensors
from tqdm import tqdm

from src.utils.helper import load_x_from_safetensor


class Audio2Exp(nn.Module):
    def __init__(self, device, sadtalker_checkpoint_path):
        super(Audio2Exp, self).__init__()
        self.device = device

        self.netG = SimpleWrapperV2()
        self.netG = self.netG.to(device)
        self.netG.eval()
        self.load_netG(sadtalker_checkpoint_path)

    def load_netG(self, sadtalker_checkpoint_path):
        checkpoints = safetensors.torch.load_file(sadtalker_checkpoint_path)
        self.netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))

    def __call__(self, batch):
        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10),'Audio2Exp Predicting...'): # every 10 frames
            
            current_mel_input = mel_input[:,i:i+10]

            #ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['pred_coeff'][:, :, :64][:, i:i+10]
            ratio = batch['blink_ratio'][:, i:i+10]                               #bs T

            audiox = current_mel_input.view(-1, 1, 80, 16)                  # bs*T 1 80 16

            curr_exp_coeff_pred  = self.netG(audiox, ref, ratio)         # bs T 64 
            exp_coeff_pred += [curr_exp_coeff_pred]

        return torch.cat(exp_coeff_pred, axis=1)



class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        
        if self.use_act:
            return self.act(out)
        else:
            return out

class SimpleWrapperV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )

        #### load the pre-trained audio_encoder 
        #self.audio_encoder = self.audio_encoder.to(device)  
        '''
        wav2lip_state_dict = torch.load('/apdcephfs_cq2/share_1290939/wenxuazhang/checkpoints/wav2lip.pth')['state_dict']
        state_dict = self.audio_encoder.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                print('init:', k)
                state_dict[k.replace('module.audio_encoder.', '')] = v
        self.audio_encoder.load_state_dict(state_dict)
        '''

        self.mapping1 = nn.Linear(512+64+1, 64)
        #self.mapping2 = nn.Linear(30, 64)
        #nn.init.constant_(self.mapping1.weight, 0.)
        nn.init.constant_(self.mapping1.bias, 0.)

    def forward(self, x, ref, ratio):
        x = self.audio_encoder(x).view(x.size(0), -1)
        ref_reshape = ref.reshape(x.size(0), -1)
        ratio = ratio.reshape(x.size(0), -1)
        
        y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1)) 
        out = y.reshape(ref.shape[0], ref.shape[1], -1) #+ ref # resudial
        return out