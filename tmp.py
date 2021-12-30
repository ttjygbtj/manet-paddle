import json


def submit_masks(param):
    pass


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
            # .transpose([2, 0, 1])
        frame_list.append(frame)
    frames = np.stack(frame_list, axis=0)
    return frames


def annotated_frames(scribbles):
    frames_list = [(i, path) for i, path in enumerate(scribbles['scribbles']) if path.size]
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


def get_images(sequence='bike-packing'):
    img_path = os.path.join('data', sequence.strip(), 'frame')
    img_files = os.listdir(img_path)
    files = []
    for img in img_files:
        img_file = np.array(Image.open(os.path.join(img_path, img)))
        files.append(img_file)
    return np.array(files)


def scribbles2mask(scribbles,
                   output_resolution,
                   seq,
                   nb_frames,
                   bresenham=True,
                   default_value=-1, ):
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
    m = masks[seq]
    for p in scribbles:
        path = p['path']
        obj_id = p['object_id']
        path = np.asarray(path, dtype=np.float)
        path *= size_array
        path = path.astype(np.int)
        if bresenham:
            path = bresenham_function(path)
        m[path[:, 0], path[:, 1]] = obj_id
    masks[seq] = m
    print(np.unique(m))
    return masks


import cv2
import os
from PIL import Image
import matplotlib.image as mpimg
import numpy as np


# image = cv2.imread('/home/lc/PaddleVideo/data/bike-packing/frame/00000.jpg')
# mask = cv2.imread('/home/lc/PaddleVideo/output/Manet_stage2/bike-packing/interactive1/turn3/00000.png')
# sss = np.zeros_like(image, dtype=np.uint8)
# for i in range(3):
#     image[:, :, i][np.where(mask[:, :, i])] = 200
# # ref = np.zeros(np.shape(image), dtype=np.uint8)
# # ref[:, :, i] = 255
# # cv2.imshow('image', ref)
# image_ = cv2.add(image, sss)
# print(image_ == image)
# cv2.imwrite('mask.png', mask)
# cv2.imwrite('sss.png', sss)
# cv2.imwrite('test.png', image)
# sequence = 'bike-packing'
# for scribbles, start_annotated_frame in get_scribbles(sequence, objects=2):
#     # print(np.unique(scribbles2mask(scribbles, start_annotated_frame )))
#     if annotated_frames(scribbles):
#         print(annotated_frames(scribbles))
def get_scribbles(file):
    # img_scribbles_path = os.path.join('data', sequence.strip(), 'obj%s')
    # img_scribbles = os.listdir(img_scribbles_path % 0)[:max_nb_interactions]
    # for img in img_scribbles:
    #     scribbles = []
    #     seq = int(os.path.splitext(img)[0].split('_')[-1])
    #     for ob in range(objects):
    #         img_scribble = np.array(Image.open(os.path.join(img_scribbles_path % ob, img)))
    #         scribbles.append({'object_id': ob + 1, 'path': np.array(
    #             np.where(
    #                 (img_scribble[:, :, 1] != img_scribble[:, :, 0]) & (img_scribble[:, :, 0] == 255))).T / np.array(
    #             img_scribble.shape[:2])})
    #     yield scribbles, seq
    with open(file) as f:
        ff = json.load(f)
    seq, paths = annotated_frames(ff)
    yield seq, paths


# with open('/home/lc/PaddleVideo/data/bike-packing/lable/1.json') as f:
with open('/home/lc/manet/data/DAVIS/Scribbles/bike-packing/003.json') as f:
    ff = json.load(f)
for i in range(7):
    with open(f'{i + 2}.json', 'r') as f:
        scribbles=json.load(f)
        1