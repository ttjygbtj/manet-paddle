from paddlevideo.utils import (build_record, log_batch, log_epoch, save, load,
                               mkdir)
from paddlevideo.loader.pipelines import ToTensor_manet
import os
import timeit
import cv2
import numpy as np
import paddle
from PIL import Image
from paddle import nn
from paddlevideo.utils.manet_utils import float_, _palette, rough_ROI, write_dict
from tools.utils import TEST
from paddlevideo.modeling.builder import build_model


def load_video(path, min_side=None):
    frame_list = []
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if min_side:
            h, w = frame.shape[:2]
            new_w = (w * min_side // min(w, h))
            new_h = (h * min_side // min(w, h))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        frame_list.append(frame)
    frames = np.stack(frame_list, axis=0)
    return frames

# TODO
def submit_masks(param):
    pass


def annotated_frames(scribbles):
    frames_list = [i for i in scribbles if i['path'].size]
    return frames_list

def bresenham_function(points):
    """ Apply Bresenham algorithm for a list points.

    More info: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm

    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the number
            if points and the second coordinate representing the (x, y)
            coordinates.

    # Returns
        ndarray: Array of points after having applied the bresenham algorithm.
    """

    points = np.asarray(points, dtype=np.int)

    def line(x0, y0, x1, y1):
        """ Bresenham line algorithm.
        """
        d_x = x1 - x0
        d_y = y1 - y0

        x_sign = 1 if d_x > 0 else -1
        y_sign = 1 if d_y > 0 else -1

        d_x = np.abs(d_x)
        d_y = np.abs(d_y)

        if d_x > d_y:
            xx, xy, yx, yy = x_sign, 0, 0, y_sign
        else:
            d_x, d_y = d_y, d_x
            xx, xy, yx, yy = 0, y_sign, x_sign, 0

        D = 2 * d_y - d_x
        y = 0

        line = np.empty((d_x + 1, 2), dtype=points.dtype)
        for x in range(d_x + 1):
            line[x] = [x0 + x * xx + y * yx, y0 + x * xy + y * yy]
            if D >= 0:
                y += 1
                D -= 2 * d_x
            D += 2 * d_y

        return line

    nb_points = len(points)
    if nb_points < 2:
        return points

    new_points = []

    for i in range(nb_points - 1):
        p = points[i:i + 2].ravel().tolist()
        new_points.append(line(*p))

    new_points = np.concatenate(new_points, axis=0)

    return new_points


def get_scribbles(sequence, max_nb_interactions=8, objects=1):
    img_scribbles_path = os.path.join('data', sequence.strip(), 'obj%s')
    img_scribbles = os.listdir(img_scribbles_path % 0)[:max_nb_interactions]
    for img in img_scribbles:
        scribbles = []
        seq = int(os.path.splitext(img)[0].split('_')[-1])
        for ob in range(objects):
            img_scribble = np.array(Image.open(os.path.join(img_scribbles_path % ob, img)))
            scribbles.append({'object_id': ob, 'path': np.array(
                np.where(
                    (img_scribble[:, :, 1] != img_scribble[:, :, 0]) & (img_scribble[:, :, 0] == 255))).T / np.array(
                img_scribble.shape[:2])})
        yield scribbles, seq


def scribbles2mask(scribbles,
                   output_resolution,
                   seq,
                   nb_frames,
                   bresenham=True,
                   default_value=-1, ):
    """ Convert the scribbles data into a mask.
    # Arguments
        scribbles: Dictionary. Scribbles in the default format.
        output_resolution: Tuple. Output resolution (H, W).
        bezier_curve_sampling: Boolean. Weather to sample first the returned
            scribbles using bezier curve or not.
        nb_points: Integer. If `bezier_curve_sampling` is `True` set the number
            of points to sample from the bezier curve.
        bresenham: Boolean. Whether to compute bresenham algorithm for the
            scribbles lines.
        default_value: Integer. Default value for the pixels which do not belong
            to any scribble.

    # Returns
        ndarray: Array with the mask of the scribbles with the index of the
            object ids. The shape of the returned array is (B x H x W) by
            default or (H x W) if `only_annotated_frame==True`.
    """
    if len(output_resolution) != 2:
        raise ValueError(
            'Invalid output resolution: {}'.format(output_resolution))
    for r in output_resolution:
        if r < 1:
            raise ValueError(
                'Invalid output resolution: {}'.format(output_resolution))

    masks = np.full(
        (nb_frames,) + output_resolution, default_value, dtype=np.int)

    size_array = np.asarray(output_resolution, dtype=np.float) - 1

    for p in scribbles:
        path = p['path']
        obj_id = p['object_id']
        path = np.asarray(path, dtype=np.float)
        path *= size_array
        path = path.astype(np.int)
        if bresenham:
            path = bresenham_function(path)
        m = masks[seq]

        m[path[:, 0], path[:, 1]] = obj_id
        masks[seq] = m

    return masks


# @paddle.no_grad()
@TEST.register()
class Manet_predict_helper(object):
    def __call__(self, weights, parallel=True, **cfg):
        # 1. Construct model.
        if cfg['MODEL'].get('backbone') and cfg['MODEL']['backbone'].get(
                'pretrained'):
            cfg['MODEL'].backbone.pretrained = ''  # disable pretrain model init
        cfg['MODEL'].head.test_mode = True
        model = build_model(cfg['MODEL'])
        if parallel:
            model = paddle.DataParallel(model)

        # 2. Construct data.
        video = 'data/1.mp4'
        sequence = 'drone'
        images = load_video(video, 480)
        # [195, 389, 238, 47, 244, 374, 175, 399]
        # .shape: (502, 480, 600, 3)
        is_save_image = True  # Save the predicted masks
        report_save_dir = cfg.get("output_dir", f"./output/{cfg['model_name']}")
        if not os.path.exists(report_save_dir):
            os.makedirs(report_save_dir)
            # Configuration used in the challenges
        max_nb_interactions = 8  # Maximum number of interactions
        max_time_per_interaction = 30  # Maximum time per interaction per object
        # Total time available to interact with a sequence and an initial set of scribbles
        max_time = max_nb_interactions * max_time_per_interaction  # Maximum time per object
        # Interactive parameters
        model.eval()

        state_dicts_ = load(weights)['state_dict']
        state_dicts = {}
        for k, v in state_dicts_.items():
            if 'num_batches_tracked' not in k:
                state_dicts['head.' + k] = v
                if ('head.' + k) not in model.state_dict().keys():
                    print(f'pretrained -----{k} -------is not in model')
        write_dict(state_dicts, 'model_for_infer.txt', **cfg)
        model.set_state_dict(state_dicts)
        inter_file = open(
            os.path.join(cfg.get("output_dir", f"./output/{cfg['model_name']}"),
                         'inter_file.txt'), 'w')
        seen_seq = False
        first_scribble = True
        with paddle.no_grad():

            t_total = timeit.default_timer()
            # Get the current iteration scribbles

            scribbles = get_scribbles()
            f, h, w = images.shape[:3]
            if 'prev_label_storage' not in locals().keys():
                prev_label_storage = paddle.zeros([f, h, w])
            if len(annotated_frames(scribbles)) == 0:
                final_masks = prev_label_storage
                submit_masks(final_masks.numpy())
                # continue

            # if no scribbles return, keep masks in previous round

            start_annotated_frame = annotated_frames(scribbles)[0]

            pred_masks = []
            pred_masks_reverse = []

            if first_scribble:  # If in the first round, initialize memories
                n_interaction = 1
                eval_global_map_tmp_dic = {}
                local_map_dics = ({}, {})
                total_frame_num = f
                obj_nums = 1
            else:
                n_interaction += 1
            inter_file.write(sequence + ' ' + 'interaction' +
                             str(n_interaction) + ' ' + 'frame' +
                             str(start_annotated_frame) + '\n')

            if first_scribble:  # if in the first round, extract pixel embbedings.
                if seen_seq:
                    inter_turn = 1

                    seen_seq = True
                    embedding_memory = []
                    places = paddle.set_device('gpu')

                    if parallel:
                        for c in model.children():
                            frame_embedding = c.head.extract_feature(
                                images)
                    else:
                        frame_embedding = model.head.extract_feature(
                            images)
                    embedding_memory.append(frame_embedding)

                    del frame_embedding

                    embedding_memory = paddle.concat(
                        embedding_memory, 0)
                    _, _, emb_h, emb_w = embedding_memory.shape
                    ref_frame_embedding = embedding_memory[
                        start_annotated_frame]
                    ref_frame_embedding = ref_frame_embedding.unsqueeze(
                        0)
                else:
                    inter_turn += 1
                    ref_frame_embedding = embedding_memory[
                        start_annotated_frame]
                    ref_frame_embedding = ref_frame_embedding.unsqueeze(
                        0)

            else:
                ref_frame_embedding = embedding_memory[
                    start_annotated_frame]
                ref_frame_embedding = ref_frame_embedding.unsqueeze(
                    0)
            ########
            scribble_masks = scribbles2mask(scribbles,
                                            (emb_h, emb_w))
            scribble_label = scribble_masks[start_annotated_frame]
            scribble_sample = {'scribble_label': scribble_label}
            scribble_sample = ToTensor_manet()(scribble_sample)
            #                     print(ref_frame_embedding, ref_frame_embedding.shape)
            scribble_label = scribble_sample['scribble_label']

            scribble_label = scribble_label.unsqueeze(0)
            model_name = cfg['model_name']
            output_dir = cfg.get("output_dir",
                                 f"./output/{model_name}")
            inter_file_path = os.path.join(
                output_dir, model_name, sequence,
                'interactive' + str(n_interaction),
                'turn' + str(inter_turn))
            if is_save_image:
                ref_scribble_to_show = scribble_label.squeeze(
                ).numpy()
                im_ = Image.fromarray(
                    ref_scribble_to_show.astype('uint8')).convert(
                    'P', )
                im_.putpalette(_palette)
                ref_img_name = str(start_annotated_frame)

                if not os.path.exists(inter_file_path):
                    os.makedirs(inter_file_path)
                im_.save(
                    os.path.join(inter_file_path,
                                 'inter_' + ref_img_name + '.png'))
            if first_scribble:
                prev_label = None
                prev_label_storage = paddle.zeros([f, h, w])
            else:
                prev_label = prev_label_storage[
                    start_annotated_frame]
                prev_label = prev_label.unsqueeze(0).unsqueeze(0)
            # check if no scribbles.
            if not first_scribble and paddle.unique(
                    scribble_label).shape[0] == 1:
                print('not first_scribble and paddle.unique(scribble_label).shape[0] == 1')
                print(paddle.unique(scribble_label))
                final_masks = prev_label_storage
                submit_masks(final_masks.numpy())
                # continue

                ###inteaction segmentation head
            if parallel:
                for c in model.children():
                    tmp_dic, local_map_dics = c.head.int_seghead(
                        ref_frame_embedding=ref_frame_embedding,
                        ref_scribble_label=scribble_label,
                        prev_round_label=prev_label,
                        global_map_tmp_dic=
                        eval_global_map_tmp_dic,
                        local_map_dics=local_map_dics,
                        interaction_num=n_interaction,
                        seq_names=[sequence],
                        gt_ids=paddle.to_tensor([obj_nums]),
                        frame_num=[start_annotated_frame],
                        first_inter=first_scribble)
            else:
                tmp_dic, local_map_dics = model.head.int_seghead(
                    ref_frame_embedding=ref_frame_embedding,
                    ref_scribble_label=scribble_label,
                    prev_round_label=prev_label,
                    global_map_tmp_dic=eval_global_map_tmp_dic,
                    local_map_dics=local_map_dics,
                    interaction_num=n_interaction,
                    seq_names=[sequence],
                    gt_ids=paddle.to_tensor([obj_nums]),
                    frame_num=[start_annotated_frame],
                    first_inter=first_scribble)
            pred_label = tmp_dic[sequence]
            pred_label = nn.functional.interpolate(
                pred_label,
                size=(h, w),
                mode='bilinear',
                align_corners=True)
            pred_label = paddle.argmax(pred_label, axis=1)
            pred_masks.append(float_(pred_label))
            prev_label_storage[start_annotated_frame] = float_(
                pred_label[0])

            if is_save_image:  # save image
                pred_label_to_save = pred_label.squeeze(
                    0).numpy()
                im = Image.fromarray(
                    pred_label_to_save.astype('uint8')).convert(
                    'P', )
                im.putpalette(_palette)
                imgname = str(start_annotated_frame)
                while len(imgname) < 5:
                    imgname = '0' + imgname
                if not os.path.exists(inter_file_path):
                    os.makedirs(inter_file_path)
                im.save(
                    os.path.join(inter_file_path,
                                 imgname + '.png'))
            #######################################
            if first_scribble:
                scribble_label = rough_ROI(scribble_label)

            ##############################
            ref_prev_label = pred_label.unsqueeze(0)
            prev_label = pred_label.unsqueeze(0)
            prev_embedding = ref_frame_embedding
            for ii in range(start_annotated_frame + 1,
                            total_frame_num):
                current_embedding = embedding_memory[ii]
                current_embedding = current_embedding.unsqueeze(
                    0)
                prev_label = prev_label
                if parallel:
                    for c in model.children():
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = c.head.prop_seghead(
                            ref_frame_embedding,
                            prev_embedding,
                            current_embedding,
                            scribble_label,
                            prev_label,
                            normalize_nearest_neighbor_distances
                            =True,
                            use_local_map=True,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            k_nearest_neighbors=cfg['TEST']
                            ['knns'],
                            global_map_tmp_dic=
                            eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            start_annotated_frame=
                            start_annotated_frame,
                            frame_num=[ii],
                            dynamic_seghead=c.head.
                                dynamic_seghead)
                else:
                    tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.head.prop_seghead(
                        ref_frame_embedding,
                        prev_embedding,
                        current_embedding,
                        scribble_label,
                        prev_label,
                        normalize_nearest_neighbor_distances=
                        True,
                        use_local_map=True,
                        seq_names=[sequence],
                        gt_ids=paddle.to_tensor([obj_nums]),
                        k_nearest_neighbors=cfg['TEST']['knns'],
                        global_map_tmp_dic=
                        eval_global_map_tmp_dic,
                        local_map_dics=local_map_dics,
                        interaction_num=n_interaction,
                        start_annotated_frame=
                        start_annotated_frame,
                        frame_num=[ii],
                        dynamic_seghead=model.dynamic_seghead)
                pred_label = tmp_dic[sequence]
                pred_label = nn.functional.interpolate(
                    pred_label,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=True)
                pred_label = paddle.argmax(pred_label, axis=1)
                pred_masks.append(float_(pred_label))
                prev_label = pred_label.unsqueeze(0)
                prev_embedding = current_embedding
                prev_label_storage[ii] = float_(pred_label[0])
                if is_save_image:
                    pred_label_to_save = pred_label.squeeze(
                        0).numpy()
                    im = Image.fromarray(
                        pred_label_to_save.astype(
                            'uint8')).convert('P', )
                    im.putpalette(_palette)
                    imgname = str(ii)
                    while len(imgname) < 5:
                        imgname = '0' + imgname
                    if not os.path.exists(inter_file_path):
                        os.makedirs(inter_file_path)
                    im.save(
                        os.path.join(inter_file_path,
                                     imgname + '.png'))
            #######################################
            prev_label = ref_prev_label
            prev_embedding = ref_frame_embedding
            #######
            # Propagation <-
            for ii in range(start_annotated_frame):
                current_frame_num = start_annotated_frame - 1 - ii
                current_embedding = embedding_memory[
                    current_frame_num]
                current_embedding = current_embedding.unsqueeze(0)
                prev_label = prev_label
                if parallel:
                    for c in model.children():
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = c.head.prop_seghead(
                            ref_frame_embedding,
                            prev_embedding,
                            current_embedding,
                            scribble_label,
                            prev_label,
                            normalize_nearest_neighbor_distances=
                            True,
                            use_local_map=True,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            k_nearest_neighbors=cfg['TEST']['knns'],
                            global_map_tmp_dic=
                            eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            start_annotated_frame=
                            start_annotated_frame,
                            frame_num=[current_frame_num],
                            dynamic_seghead=c.head.dynamic_seghead)
                else:
                    tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.head.prop_seghead(
                        ref_frame_embedding,
                        prev_embedding,
                        current_embedding,
                        scribble_label,
                        prev_label,
                        normalize_nearest_neighbor_distances=True,
                        use_local_map=True,
                        seq_names=[sequence],
                        gt_ids=paddle.to_tensor([obj_nums]),
                        k_nearest_neighbors=cfg['TEST']['knns'],
                        global_map_tmp_dic=eval_global_map_tmp_dic,
                        local_map_dics=local_map_dics,
                        interaction_num=n_interaction,
                        start_annotated_frame=start_annotated_frame,
                        frame_num=[current_frame_num],
                        dynamic_seghead=model.head.dynamic_seghead)
                pred_label = tmp_dic[sequence]
                pred_label = nn.functional.interpolate(
                    pred_label,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=True)

                pred_label = paddle.argmax(pred_label, axis=1)
                pred_masks_reverse.append(float_(pred_label))
                prev_label = pred_label.unsqueeze(0)
                prev_embedding = current_embedding
                ####
                prev_label_storage[current_frame_num] = float_(
                    pred_label[0])
                ###
                if is_save_image:
                    pred_label_to_save = pred_label.squeeze(
                        0).numpy()
                    im = Image.fromarray(
                        pred_label_to_save.astype('uint8')).convert(
                        'P', )
                    im.putpalette(_palette)
                    imgname = str(current_frame_num)
                    while len(imgname) < 5:
                        imgname = '0' + imgname
                    if not os.path.exists(inter_file_path):
                        os.makedirs(inter_file_path)
                    im.save(
                        os.path.join(inter_file_path,
                                     imgname + '.png'))
            pred_masks_reverse.reverse()
            pred_masks_reverse.extend(pred_masks)
            final_masks = paddle.concat(pred_masks_reverse, 0)
            submit_masks(final_masks.numpy())

            t_end = timeit.default_timer()
            print('Total time for single interaction: ' +
                  str(t_end - t_total))
            first_scribble = False

        inter_file.close()
