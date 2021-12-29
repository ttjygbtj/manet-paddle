label_tmp = label_tmp.transpose([1, 2, 0])
label = (float_(label_tmp) == float_(obj_ids))
label = label.unsqueeze(-1).transpose([3, 2, 0, 1])
label_dic[seq_] = float_(label)

for seq_ in tmp_dic.keys():
    tmp_pred_logits = tmp_dic[seq_]
    tmp_pred_logits = nn.functional.interpolate(tmp_pred_logits, size=(h, w), mode='bilinear',
                                                align_corners=True)
    tmp_dic[seq_] = tmp_pred_logits

    label_tmp, obj_num = label_and_obj_dic[seq_]
    obj_ids = np.arange(1, obj_num + 1)
    obj_ids = paddle.to_tensor(obj_ids)
    obj_ids = int_(obj_ids)