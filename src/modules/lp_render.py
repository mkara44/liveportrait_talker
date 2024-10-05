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


    def make_motion_template_from_pred(self, pred, pitch_weight, yaw_weight, roll_weight):
        bs = pred['exp'].shape[0]
        #pred['pitch'] = headpose_pred_to_degree(pred['pitch'])[:, None] * pitch_weight  # Bx1
        #pred['yaw'] = headpose_pred_to_degree(pred['yaw'])[:, None] * yaw_weight  # Bx1
        #pred['roll'] = headpose_pred_to_degree(pred['roll'])[:, None] * roll_weight  # Bx1
        pred['exp'] = pred['exp'].reshape(bs, -1, 3)  # BxNx3

        #R_i = get_rotation_matrix(pred['pitch'], pred['yaw'], pred['roll'])
        item_dct = {#'R': R_i.to(self.device),
                    'exp': pred['exp'].to(self.device),
                    #'t': pred['t'].to(self.device),
                    }

        template_dct = {'motion': item_dct}
        return template_dct
    
    def __call__(self, batch):
        n_frames = batch["num_frames"]
        source_type = batch["source_type"]
        source_eye_close_ratio = batch["source_eye_close_ratio"]
        driving_blink_ratio = batch["liveportrait_blink_ratio"]

        x_s_info = self.live_portrait_wrapper.get_kp_info(batch["rendering_input_face"].to(self.device))
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])


        f_s = self.live_portrait_wrapper.extract_feature_3d(batch["rendering_input_face"].to(self.device))
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        driving_template_dct = self.make_motion_template_from_pred(pred=batch["mapped_semantics"].copy(),
                                                                   pitch_weight=batch["pitch_weight"],
                                                                   yaw_weight=batch["yaw_weight"],
                                                                   roll_weight=batch["roll_weight"])

        x_d_i_info = driving_template_dct["motion"]
        delta_new = x_s_info['exp'].clone()

        I_p_lst = []
        for i in tqdm(range(n_frames), "Rendering.."):
            delta_new = x_s_info['exp'].clone()
            R_new = R_s #if batch["source_type"] == "video" else x_d_i_info["R"][i]

            for lip_idx in [6, 12, 14, 17, 19, 20]:
                delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'][i] - x_d_i_info['exp'][0]))[:, lip_idx, :]

            scale_new = x_s_info['scale']
            t_new = x_s_info['t']
            t_new[..., 2].fill_(0)

            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if source_type == "image":
                combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(driving_blink_ratio[i], source_eye_close_ratio[i])
                eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                x_d_i_new += eyes_delta if eyes_delta is not None else 0

            x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            x_d_i_new = x_s + (x_d_i_new - x_s) * self.liveportrait_cfg.driving_multiplier

            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_lst.append(out["out"])    

        batch["rendered_frame_list"] = I_p_lst            
        return batch