import torch
import numpy as np
from tqdm import tqdm

from src.utils.lp_render.helper import load_lip_array, get_reference_frames, smooth, retarget_pupil
from src.utils.lp_render.wrapper import LivePortraitWrapper
from src.utils.lp_render.camera import get_rotation_matrix
from src.utils.lp_render.synthetic_headpose_generation import SyntheticHeadPoseGeneration


class LivePortraitRender:
    lip_idx_list = [6, 12, 14, 17, 19, 20]

    def __init__(self, device, fps, instant_save_func, driving_smooth_observation_variance, liveportrait_cfg, synthetic_headpose_generation_cfg):
        self.device = device
        self.fps = fps
        self.driving_smooth_observation_variance = driving_smooth_observation_variance
        self.liveportrait_cfg = liveportrait_cfg
        self.instant_save_func = instant_save_func

        self.lip_array = torch.from_numpy(load_lip_array(self.liveportrait_cfg.lip_array_pkl_path)).to(dtype=torch.float32, device=device)
        self.synthetic_headpose_generation = SyntheticHeadPoseGeneration(device=self.device,
                                                                         fps=self.fps,
                                                                         **synthetic_headpose_generation_cfg)
        self.live_portrait_wrapper = LivePortraitWrapper(device=device,
                                                         liveportrait_cfg=liveportrait_cfg)


    def make_motion_template_from_pred(self, pred):
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
    
    def make_ref_head_pose_template(self, ref_rendering_input_face):
        ref_R_list = []
        
        for i in tqdm(range(ref_rendering_input_face.shape[0]), desc="Generating reference head poses..."):
            ref_i_info = self.live_portrait_wrapper.get_kp_info(ref_rendering_input_face[i].unsqueeze(0))
            R_i = get_rotation_matrix(ref_i_info['pitch'], ref_i_info['yaw'], ref_i_info['roll'])

            ref_R_list.append(R_i)
        return ref_R_list
    
    def predict_source_inputs(self, rendering_input_face):
        x_s_i_info = self.live_portrait_wrapper.get_kp_info(rendering_input_face.to(self.device))
        R_s_i = get_rotation_matrix(x_s_i_info['pitch'], x_s_i_info['yaw'], x_s_i_info['roll'])

        f_s_i = self.live_portrait_wrapper.extract_feature_3d(rendering_input_face.to(self.device))
        x_s_i = self.live_portrait_wrapper.transform_keypoint(x_s_i_info)

        return x_s_i_info, R_s_i, f_s_i, x_s_i

    def __call__(self, batch):
        n_frames = batch["num_frames"]
        still = batch["still"]
        pupil_x = batch["pupil_x"]
        pupil_y = batch["pupil_y"]
        source_type = batch["source_type"]
        source_coeff = batch["source_coeff"]
        rendering_input_face = batch["rendering_input_face"]
        source_eye_close_ratio = batch.get("source_eye_close_ratio", None)
        driving_blink_ratio = batch.get("liveportrait_blink_ratio", None)

        ref_head_pose_path = batch["ref_head_pose_path"]
        ref_frames_from_zero = batch["ref_frames_from_zero"]
        ref_R_list = batch.get("ref_R_list", None)
        ref_rendering_input_face = batch.get("ref_rendering_input_face", None)
     
        driving_template_dct = self.make_motion_template_from_pred(pred=batch["mapped_semantics"].copy())
        
        if ref_head_pose_path is not None:
            if ref_R_list is None:
                ref_R_list = self.make_ref_head_pose_template(ref_rendering_input_face.to(self.device))
                batch["ref_R_list"] = ref_R_list

            ref_R_list = get_reference_frames(ref_R_list=ref_R_list, n_frames=n_frames, ref_frames_from_zero=ref_frames_from_zero)

        
        x_d_exp_lst_smooth = None
        rendered_frame_list = []
        x_d_i_info = driving_template_dct["motion"]
        for i in tqdm(range(n_frames), "Rendering.."):
            if source_type == "video":
                x_s_i_info, R_s_i, f_s_i, x_s_i = self.predict_source_inputs(rendering_input_face=rendering_input_face[i].unsqueeze(0))

                if x_d_exp_lst_smooth is None and i == 0:
                    x_d_exp_lst = [x_s_i_info['exp'][:, self.lip_idx_list, :].detach().cpu().numpy() + driving_template_dct['motion']['exp'][i].detach().cpu().numpy() - driving_template_dct['motion']['exp'][0].detach().cpu().numpy() for i in range(n_frames)]
                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, x_s_i_info['exp'][:, self.lip_idx_list, :].shape, self.device, self.driving_smooth_observation_variance)

            elif source_type == "image" and i == 0:
                x_s_i_info, R_s_i, f_s_i, x_s_i = self.predict_source_inputs(rendering_input_face=rendering_input_face)
                #self.instant_save_func(source_type=source_type, source_coeff=source_coeff, source_eye_close_ratio=source_eye_close_ratio, x_s_i_info=x_s_i_info, R_s_i=R_s_i, f_s_i=f_s_i, x_s_i=x_s_i)

                if not still:
                    if ref_R_list is None:
                        ref_R_list = self.synthetic_headpose_generation(num_frames=n_frames,
                                                                        lower_upper_lip_expressions=driving_template_dct["motion"]["exp"][:, [19,20], :],
                                                                        source_info=x_s_i_info)

                    else:
                        ref_R_list = [R.detach().cpu().numpy() for R in ref_R_list]
                        ref_R_list = smooth(ref_R_list, R_s_i.shape, self.device, self.driving_smooth_observation_variance)

            x_c_s_i = x_s_i_info["kp"]
            delta_new = x_s_i_info['exp'].clone()
            R_new = R_s_i if ref_R_list is None else ref_R_list[i]

            for idx, lip_idx in enumerate(self.lip_idx_list):
                delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][idx, :] if source_type == "video" else (x_s_i_info['exp'][:, lip_idx, :] + (x_d_i_info['exp'][i] - x_d_i_info['exp'][0])[idx, :])

            if pupil_x != 0 or pupil_y != 0:
                delta_new = retarget_pupil(delta_new=delta_new, pupil_x=pupil_x, pupil_y=pupil_y)

            scale_new = x_s_i_info['scale']
            t_new = x_s_i_info['t']
            t_new[..., 2].fill_(0)

            x_d_i_new = scale_new * (x_c_s_i @ R_new + delta_new) + t_new


            if source_type == "image":
                combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(driving_blink_ratio[i], source_eye_close_ratio[i])
                eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s_i, combined_eye_ratio_tensor)
                x_d_i_new += eyes_delta if eyes_delta is not None else 0

            x_d_i_new = self.live_portrait_wrapper.stitching(x_s_i, x_d_i_new)
            x_d_i_new = x_s_i + (x_d_i_new - x_s_i) * self.liveportrait_cfg.driving_multiplier

            out = self.live_portrait_wrapper.warp_decode(f_s_i, x_s_i, x_d_i_new)
            rendered_frame_list.append(out["out"])

        batch["rendered_frame_list"] = rendered_frame_list            
        return batch