import cv2
import onnxruntime
import os
import pdb
import math
import numpy as np

def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded

def cal_bbox(score_map_ctr, size_map, offset_map):

    # max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
    # feat_sz = 16
    # print("score: ", max_score.item())
    max_conf = 0

    idx_x = 0
    idx_y = 0


    for i in range(16): #h 
        for j in range(16): # w
            cur_conf = score_map_ctr[0][0][i][j]

            if (cur_conf > max_conf):
                max_conf = cur_conf
                idx_y = i
                idx_x = j


    # if max_score < conf_threshold:
    #     return False
    # idx_y1 = idx // feat_sz
    # idx_x1 = idx % feat_sz

    

    # feat_sz = score_map_ctr.shape[2]
    # max_index_flat = np.argmax(score_map_ctr[0][0])
    # idx_x,idx_y  = np.unravel_index(max_index_flat, score_map_ctr[0][0].shape)

    w = size_map[0][0][idx_x][idx_y]
    h = size_map[0][1][idx_x][idx_y]

    offset_x = offset_map[0][0][idx_x][idx_y]
    offset_y = offset_map[0][1][idx_x][idx_y]

    bbox = np.column_stack([(idx_x + offset_x) / feat_sz,
                            (idx_y + offset_y) / feat_sz,
                            w, h])

    print("score: ", max_conf, "index x: ", idx_x, idx_y, w,h, offset_x, offset_y)

    #pdb.set_trace()

    return bbox

# def cal_bbox(score_map_ctr, size_map, offset_map, return_score=False, conf_threshold=0.01):
#     max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
#     feat_sz = 16
#     print("score: ", max_score.item())
#     # if max_score < conf_threshold:
#     #     return False
#     idx_y = idx // feat_sz
#     idx_x = idx % feat_sz


#     idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
#     size = size_map.flatten(2).gather(dim=2, index=idx)
#     offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

#     # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
#     #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
#     # cx, cy, w, h
#     bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / feat_sz,
#                       (idx_y.to(torch.float) + offset[:, 1:]) / feat_sz,
#                       size.squeeze(-1)], dim=1)

#     #pdb.set_trace()

#     if return_score:
#         return bbox, max_score

#     return bbox

def map_box_back(pred_box: list, resize_factor: float, state: list):
    cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * 256 / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]


# def hann1d(sz: int, centered = True) -> torch.Tensor:
#     """1D cosine window."""
#     if centered:
#         return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
#     w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz//2 + 1).float()))
#     return torch.cat([w, w[1:sz-sz//2].flip((0,))])


# def hann2d(sz: torch.Tensor, centered = True) -> torch.Tensor:
#     """2D cosine window."""
#     return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)

if __name__ == "__main__":
    videofilepath = 'bag.avi'
    os.makedirs("vis", exist_ok=1)
    init_bbox = [316, 138, 110, 118]

    feat_sz = 16
    search_size = 256
    #output_window = hann2d(torch.tensor([feat_sz, feat_sz]).long(), centered=True)
    cap = cv2.VideoCapture(videofilepath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (480, 360))


    success, frame = cap.read()
    H, W, _ = frame.shape

    z_patch_arr, _, z_amask_arr = sample_target(frame, init_bbox, 2, output_sz=128)

    print(z_patch_arr.shape)

    cv2.imwrite("test_tmeplate.jpg", z_patch_arr)

    template = z_patch_arr.transpose(2,0,1)[None,:,:,:]/ 255.

    w = 'sim_cnn_track.onnx' #文件名 请自行修改
    print("--> \n onnx file: ", w)
    providers =  ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(w, providers=providers)

    state = init_bbox
    count = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        img_255 = frame.copy()
        x_patch_arr, resize_factor, x_amask_arr = sample_target(frame, state, 4, output_sz=256)
        x_patch_arr_cp = x_patch_arr.copy()
        search = x_patch_arr.transpose(2,0,1)[None,:,:,:] / 255.
        out_dict = {}
        ######################################################################################################################
        #with torch.no_grad():
        outs = session.run(None,{"z": template.astype('float32'), "x":search.astype('float32')},)

        score_map = outs[-3]# * output_window#.cpu().numpy()
        size_map = outs[-2]
        offset_map = outs[-1]

        pred_boxes = cal_bbox(score_map,  size_map,  offset_map)
        
        # pred_boxes = pred_boxes.view(-1, 4)
        print("search_size: ", search_size)
        print("resize_factor: ", resize_factor)
        pred_box = (pred_boxes * search_size / resize_factor).tolist()[0]  # (cx, cy, w, h) 


        print(search_size, resize_factor, pred_box)

        #pdb.set_trace()
        # state = list(map(int, clip_box(map_box_back(pred_box, resize_factor, state), H, W, margin=10)))
        state = list(map(int, map_box_back(pred_box, resize_factor, state)))

        print("state: ", state)

       # pdb.set_trace()
        cv2.rectangle(img_255, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

        #print(state)
        out.write(img_255)
        cv2.imwrite("vis/%d.jpg"%count, img_255)
        
        count += 1


       # break

    out.release()
    cap.release()