def load_x_from_safetensor(checkpoint, key):
    x_generator = {}
    for k,v in checkpoint.items():
        if key in k:
            x_generator[k.replace(key+'.', '')] = v
    return x_generator

def transform_semantic_target(coeff_3dmm, frame_index, semantic_radius):
    num_frames = coeff_3dmm.shape[0]
    seq = list(range(frame_index- semantic_radius, frame_index + semantic_radius))
    index = [min(max(item, 0), num_frames-1) for item in seq ] 
    coeff_3dmm_g = coeff_3dmm[index, :]
    return coeff_3dmm_g.transpose(1,0)