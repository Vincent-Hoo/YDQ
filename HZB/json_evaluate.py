import argparse
import json_tricks as json
import os
import numpy as np
from collections import OrderedDict

def parse_arg():
    parser = argparse.ArgumentParser(description="Evaluating model")
    parser.add_argument('--pred_path', type=str, help="prediction results")
    parser.add_argument('--gt_path', type=str, help="prediction results")
    args = parser.parse_args()
    return args

def convert_json_to_array(json_file, key):
    t = []
    for f in json_file:
        t.append(f[key])
    t = np.array(t)
    if t.ndim == 3:
        return np.transpose(t, [1, 2, 0])
    else:
        return np.transpose(t, [1, 0])

def evaluate(preds, gt):
    pos_pred_src = convert_json_to_array(preds, "pred")
    jnt_visible = convert_json_to_array(gt, "joints_vis")
    pos_gt_src = convert_json_to_array(gt, "joints")
    headboxes_src = convert_json_to_array(gt, "headboxes_src")

    SC_BIAS = 0.6
    threshold = 0.5

    head = 9
    lsho = 13
    lelb = 14
    lwri = 15
    lhip = 3
    lkne = 4
    lank = 5

    rsho = 12
    relb = 11
    rwri = 10
    rkne = 1
    rank = 0
    rhip = 2

    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                        jnt_visible)
    PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)
    
    rng = np.arange(0, 0.5+0.01, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                            jnt_visible)
        pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                    jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True
    

    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_count.mask[6:8] = True
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
    

    name_value = [
        ('Head', PCKh[head]),
        ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
        ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
        ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
        ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
        ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
        ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
        ('Mean', np.sum(PCKh * jnt_ratio)),
        ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
    ]
    name_value = OrderedDict(name_value)

    return name_value, name_value['Mean']

def _print_name_value(name_values):
    for k, v in name_values.items():
        print("{}: {}".format(k, v))


def main():
    args = parse_arg()
    pred = json.load(args.pred_path)
    gt = json.load(args.gt_path)

    name_values, mean_PCKh = evaluate(pred, gt)
    _print_name_value(name_values)


if __name__ == "__main__":
    main()


