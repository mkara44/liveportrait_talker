import torch
from tqdm import tqdm

from src.utils.lp_render.helper import load_lip_array, calc_motion_multiplier
from src.utils.lp_render.wrapper import LivePortraitWrapper
from src.utils.lp_render.camera import get_rotation_matrix, headpose_pred_to_degree


class LivePortraitRender:
    def __init__(self, device, liveportrait_cfg):
        self.device = device
        self.liveportrait_cfg = liveportrait_cfg

        self.lip_array = torch.from_numpy(load_lip_array(self.liveportrait_cfg.lip_array_pkl_path)).to(dtype=torch.float32, device=device)
        self.live_portrait_wrapper = LivePortraitWrapper(device=device,
                                                         liveportrait_cfg=liveportrait_cfg)


    def make_motion_template_from_pred(self, pred, head_pose_weight):
        bs = pred['exp'].shape[0]
        pred['pitch'] = headpose_pred_to_degree(pred['pitch'])[:, None] * head_pose_weight  # Bx1
        pred['yaw'] = headpose_pred_to_degree(pred['yaw'])[:, None] * head_pose_weight  # Bx1
        pred['roll'] = headpose_pred_to_degree(pred['roll'])[:, None] * head_pose_weight  # Bx1
        pred['exp'] = pred['exp'].reshape(bs, -1, 3)  # BxNx3

        R_i = get_rotation_matrix(pred['pitch'], pred['yaw'], pred['roll'])

        item_dct = {'scale': pred['scale'].to(self.device),
                    'R': R_i.to(self.device),
                    'exp': pred['exp'].to(self.device),
                    't': pred['t'].to(self.device),}

        template_dct = {'motion': item_dct}
        return template_dct
    
    def __call__(self, batch):
        n_frames = batch["num_frames"]

        x_s_info = self.live_portrait_wrapper.get_kp_info(batch["rendering_input_face"].to(self.device))
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])

        f_s = self.live_portrait_wrapper.extract_feature_3d(batch["rendering_input_face"].to(self.device))
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        driving_template_dct = self.make_motion_template_from_pred(batch["mapped_semantics"].copy(), batch["head_pose_weight"])

        x_d_i_info = driving_template_dct["motion"]
        delta_new = x_s_info['exp'].clone()

        I_p_lst = []
        for i in tqdm(range(n_frames), "Rendering.."):
            R_d_i = x_d_i_info['R']

            if i == 0:  # cache the first frame
                R_d_0 = R_d_i[0]
                x_d_0_info = x_d_i_info.copy()

            delta_new = x_s_info['exp'].clone()
            R_new = (R_d_i[i].unsqueeze(0) @ R_d_0.unsqueeze(0).permute(0, 2, 1)) @ R_s

            #delta_new = x_s_info['exp'] + (x_d_i_info['exp'][i] - x_d_0_info['exp'][0])
            for lip_idx in [6, 12, 14, 17, 19, 20]:
                delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'][i] - x_d_0_info['exp'][0]))[:, lip_idx, :]

            #for eyes_idx in [11, 13, 15, 16, 18]:
            #    delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'][i] - x_d_0_info['exp'][0]))[:, eyes_idx, :]

            scale_new = x_s_info['scale'] #* (x_d_i_info['scale'][i] / x_d_0_info['scale'][0]) #x_s_info['scale']
            t_new = x_s_info['t'] + (x_d_i_info['t'][i] - x_d_0_info['t'][0]) #x_s_info['t']
            t_new[..., 2].fill_(0)
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
            
            if i == 0:
                x_d_0_new = x_d_i_new
                motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
            x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
            x_d_i_new = x_d_diff + x_s
            x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            
            x_d_i_new = x_s + (x_d_i_new - x_s) * self.liveportrait_cfg.driving_multiplier
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)

            I_p_lst.append(out["out"])    

        batch["rendered_frame_list"] = I_p_lst            
        return batch