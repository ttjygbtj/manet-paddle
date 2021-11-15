import os
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn import init
import cv2
import sys

from utils.api import kaiming_normal_

sys.path.append("..")
from networks.sync_batchnorm import SynchronizedBatchNorm2d
from config import cfg
import time
import paddle.nn.functional as F

#############################################################GLOBAL_DIST_MAP
USE_CORRELATION_COST = False
MODEL_UNFOLD=True
WRONG_LABEL_PADDING_DISTANCE = 1e20
if USE_CORRELATION_COST:
    from spatial_correlation_sampler import SpatialCorrelationSampler



def _pairwise_distances(x,y,ys=None):
    """Computes pairwise squared l2 distances between tensors x and y.
    Args:
    x: Tensor of shape [n, feature_dim].
    y: Tensor of shape [m, feature_dim].
    Returns:
    Float32 distances tensor of shape [n, m].
    """

    xs = paddle.sum(x*x,1)
    xs = xs.unsqueeze(1)
    if ys is None:
        ys = paddle.sum(y*y,1)
        ys = ys.unsqueeze(0)
    else:
        ys = ys
    d = xs+ys -2.*paddle.matmul(x,paddle.t(y))
    return d,ys

##################
def _flattened_pairwise_distances(reference_embeddings, query_embeddings,ys):
    """Calculates flattened tensor of pairwise distances between ref and query.
    Args:
    reference_embeddings: Tensor of shape [..., embedding_dim],
      the embedding vectors for the reference frame
    query_embeddings: Tensor of shape [n_query_images, height, width,
      embedding_dim], the embedding vectors for the query frames.
    Returns:
    A distance tensor of shape [reference_embeddings.size / embedding_dim,
    query_embeddings.size / embedding_dim]
    """
    embedding_dim = query_embeddings.size()[-1]
    reference_embeddings = reference_embeddings.view(-1, embedding_dim)
    first_dim = -1
    query_embeddings = query_embeddings.view(first_dim,embedding_dim)
    dists,ys = _pairwise_distances(query_embeddings, reference_embeddings,ys)
    return dists,ys


def _nn_features_per_object_for_chunk(
    reference_embeddings, query_embeddings, wrong_label_mask,
    k_nearest_neighbors,ys):
    """Extracts features for each object using nearest neighbor attention.
  Args:
    reference_embeddings: Tensor of shape [n_chunk, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings: Tensor of shape [m_chunk, embedding_dim], the embedding
      vectors for the query frames.
    wrong_label_mask:
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
  Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [m_chunk, n_objects, feature_dim].
    """
#    reference_embeddings_key = reference_embeddings
#    query_embeddings_key = query_embeddings
    dists,ys = _flattened_pairwise_distances(reference_embeddings,query_embeddings,ys)
    
    dists = (paddle.unsqueeze(dists,1) +
            paddle.unsqueeze(wrong_label_mask.float(),0) *
           WRONG_LABEL_PADDING_DISTANCE)
    if k_nearest_neighbors == 1:
        features,_= paddle.min(dists,2,keepdim=True)
    else:
        dists,_=paddle.topk(-dists,k=k_nearest_neighbors,dim=2)
        dists = -dists
        valid_mask = (dists<WRONG_LABEL_PADDING_DISTANCE)
        masked_dists = dists * valid_mask.float()
        pad_dist = paddle.max(masked_dists, dim=2,keepdim=True)[0].repeat((1,1,masked_dists.size()[-1]))
        dists = paddle.where(valid_mask, dists, pad_dist)
         # take mean of distances
        features = paddle.mean(dists, dim=2, keepdim=True)


    return features,ys

###
def _selected_pixel(ref_labels_flat,ref_emb_flat):
    index_list = paddle.arange(len(ref_labels_flat))
    index_list=index_list.cuda()
    index_ = paddle.masked_select(index_list,ref_labels_flat!=-1)
    
    index_=index_.long()
    ref_labels_flat=paddle.index_select(ref_labels_flat,0,index_)
    ref_emb_flat=paddle.index_select(ref_emb_flat,0,index_)

    return ref_labels_flat,ref_emb_flat
###


def _nearest_neighbor_features_per_object_in_chunks(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
    ref_obj_ids, k_nearest_neighbors, n_chunks):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
    reference_embeddings_flat: Tensor of shape [n, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings_flat: Tensor of shape [m, embedding_dim], the embedding
      vectors for the query frames.
    reference_labels_flat: Tensor of shape [n], the class labels of the
      reference frame.
    ref_obj_ids: int tensor of unique object ids in the reference labels.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    n_chunks: Integer, the number of chunks to use to save memory
      (set to 1 for no chunking).
    Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [m, n_objects, feature_dim].
    """
    chunk_size = int(np.ceil(float(query_embeddings_flat.size()[0])
                                        / n_chunks))
    if cfg.TEST_MODE:
        reference_labels_flat,reference_embeddings_flat=_selected_pixel(reference_labels_flat,reference_embeddings_flat)
    wrong_label_mask = (reference_labels_flat!=paddle.unsqueeze(ref_obj_ids,1))
    all_features = []
    for n in range(n_chunks):
        if n==0:
            ys=None
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        features,ys = _nn_features_per_object_for_chunk(
            reference_embeddings_flat, query_embeddings_flat_chunk,
            wrong_label_mask, k_nearest_neighbors,ys
            )
        all_features.append(features)
    if n_chunks == 1:
        nn_features = all_features[0]
    else:
        nn_features = paddle.cat(all_features, dim=0)
    return nn_features


def nearest_neighbor_features_per_object(
    reference_embeddings, query_embeddings, reference_labels,
     k_nearest_neighbors, gt_ids=None, n_chunks=100):
    """Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
    reference_embeddings: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings: Tensor of shape [n_query_images, height, width,
      embedding_dim], the embedding vectors for the query frames.
    reference_labels: Tensor of shape [height, width, 1], the class labels of
      the reference frame.
    max_neighbors_per_object: Integer, the maximum number of candidates
      for the nearest neighbor query per object after subsampling,
      or 0 for no subsampling.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    gt_ids: Int tensor of shape [n_objs] of the sorted unique ground truth
      ids in the first frame. If None, it will be derived from
      reference_labels.
    n_chunks: Integer, the number of chunks to use to save memory
      (set to 1 for no chunking).
    Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [n_query_images, height, width, n_objects, feature_dim].
    gt_ids: An int32 tensor of the unique sorted object ids present
      in the reference labels.
    """
    
    assert (reference_embeddings.size()[:2]==reference_labels.size()[:2])
    h,w,_ =query_embeddings.size()
    reference_labels_flat = reference_labels.view(-1)
    if gt_ids is None:
        ref_obj_ids = paddle.unique(reference_labels_flat)[-1]
        ref_obj_ids = np.arange(0,ref_obj_ids.cpu()+1)
        gt_ids = paddle.from_numpy(ref_obj_ids)
        gt_ids = gt_ids.int()
        if paddle.cuda.is_available():
            gt_ids = gt_ids.cuda()
    else:
        gt_ids = paddle.arange(0,gt_ids+1).cuda().int()

    embedding_dim = query_embeddings.size()[-1]
    query_embeddings_flat = query_embeddings.view(-1,embedding_dim)
    reference_embeddings_flat = reference_embeddings.view(-1,embedding_dim)
    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        gt_ids, k_nearest_neighbors, n_chunks)
    nn_features_dim = nn_features.size()[-1]
    nn_features=nn_features.view(1,h,w,gt_ids.size(0),nn_features_dim)
    return nn_features, gt_ids
########################################################################LOCAL_DIST_MAP
def local_pairwise_distances(x, y, max_distance=9):
    """Computes pairwise squared l2 distances using a local search window.
    Optimized implementation using correlation_cost.
    Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.
    Returns:
    Float32 distances tensor of shape
      [height, width, (2 * max_distance + 1) ** 2].

    """
    if cfg.MODEL_LOCAL_DOWNSAMPLE:
            #####
        ori_h,ori_w,_=x.size()
        x = x.permute(2,0,1).unsqueeze(0)

        x = F.avg_pool2d(x,(2,2),(2,2))
        y= y.permute(2,0,1).unsqueeze(0)
        y = F.avg_pool2d(y,(2,2),(2,2))

        x =x.squeeze(0).permute(1,2,0)
        y = y.squeeze(0).permute(1,2,0)
        corr = cross_correlate(x, y, max_distance=max_distance)
        xs = paddle.sum(x*x,2,keepdim=True)

        ys = paddle.sum(y*y,2,keepdim=True)
        ones_ys = paddle.ones_like(ys)
        ys = cross_correlate(ones_ys, ys, max_distance=max_distance)
        d = xs + ys - 2 * corr
    # Boundary should be set to Inf.
        tmp = paddle.zeros_like(d)
        boundary = paddle.eq(
            cross_correlate(ones_ys, ones_ys, max_distance=max_distance), 0)
        d = paddle.where(boundary, tmp.fill_(float('inf')),d)
        d = (paddle.sigmoid(d)-0.5)*2
        d = d.permute(2,0,1).unsqueeze(0)
        d = F.interpolate(d,size=(ori_h,ori_w),mode='bilinear',align_corners=True)
        d = d.squeeze(0).permute(1,2,0)
    else:
        corr = cross_correlate(x, y, max_distance=max_distance)
        xs = paddle.sum(x*x,2,keepdim=True)

        ys = paddle.sum(y*y,2,keepdim=True)
        ones_ys = paddle.ones_like(ys)
        ys = cross_correlate(ones_ys, ys, max_distance=max_distance)
        d = xs + ys - 2 * corr
        # Boundary should be set to Inf.
        tmp = paddle.zeros_like(d)
        boundary = paddle.eq(
            cross_correlate(ones_ys, ones_ys, max_distance=max_distance), 0)
        d = paddle.where(boundary, tmp.fill_(float('inf')),d)
    return d
def local_pairwise_distances2(x, y, max_distance=9):
    """Computes pairwise squared l2 distances using a local search window.
    Naive implementation using map_fn.
    Used as a slow fallback for when correlation_cost is not available.
    Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.
    Returns:
    Float32 distances tensor of shape
      [height, width, (2 * max_distance + 1) ** 2].
    """
    if cfg.MODEL_LOCAL_DOWNSAMPLE:
        ori_h,ori_w,_ = x.size()
        x = x.permute(2,0,1).unsqueeze(0)
        x = F.avg_pool2d(x,(2,2),(2,2))
        y = y.permute(2,0,1).unsqueeze(0)
        y = F.avg_pool2d(y,(2,2),(2,2))

        _,channels,height,width=x.size()
        padding_val=1e20
        padded_y = F.pad(y, (max_distance, max_distance,
            max_distance, max_distance), mode='constant', value=padding_val)
        offset_y = F.unfold(padded_y, kernel_size=(height, width)).view(1, channels, height, width, -1)
        x = x.view(1, channels, height, width, 1)
        minus = x - offset_y
        dists = paddle.sum(paddle.mul(minus, minus), dim=1).view(1, height, width, -1).permute(0, 3, 1, 2)
        dists = (paddle.sigmoid(dists) - 0.5) * 2
        dists = F.interpolate(dists, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
        dists = dists.squeeze(0).permute(1, 2, 0)


    else:
        padding_val = 1e20
        padded_y =nn.functional.pad(y,(0,0,max_distance, max_distance,
                              max_distance, max_distance),mode='constant', value=padding_val)
        height, width, _ = x.size()
        dists = []
        for y_start in range(2 * max_distance + 1):
            y_end = y_start + height
            y_slice = padded_y[y_start:y_end]
            for x_start in range(2 * max_distance + 1):
                x_end = x_start + width
                offset_y = y_slice[:, x_start:x_end]
                dist = paddle.sum(paddle.pow((x-offset_y),2), dim=2)
                dists.append(dist)
        dists = paddle.stack(dists, dim=2)

    return dists


def cross_correlate(x, y, max_distance=9):
    """Efficiently computes the cross correlation of x and y.
  Optimized implementation using correlation_cost.
  Note that we do not normalize by the feature dimension.
  Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.
  Returns:
    Float32 tensor of shape [height, width, (2 * max_distance + 1) ** 2].
    """
    corr_op=SpatialCorrelationSampler(kernel_size=1, patch_size=2*max_distance+1, stride=1,dilation_patch=1,padding=0)

    xs = x.permute(2,0,1)
    xs = paddle.unsqueeze(xs,0)
    ys = y.permute(2,0,1)
    ys = paddle.unsqueeze(ys,0)
    corr = corr_op(xs,ys)
    bs,_,_,hh,ww = corr.size()
    corr = corr.view(bs,-1,hh,ww)
    corr = paddle.squeeze(corr,0)
    corr = corr.permute(1,2,0)
    return corr



def local_previous_frame_nearest_neighbor_features_per_object(
    prev_frame_embedding, query_embedding, prev_frame_labels,
      gt_ids, max_distance=12):
    """Computes nearest neighbor features while only allowing local matches.
  Args:
    prev_frame_embedding: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the last frame.
    query_embedding: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the query frames.
    prev_frame_labels: Tensor of shape [height, width, 1], the class labels of
      the previous frame.
    gt_ids: Int Tensor of shape [n_objs] of the sorted unique ground truth
      ids in the first frame.
    max_distance: Integer, the maximum distance allowed for local matching.
  Returns:
    nn_features: A float32 np.array of nearest neighbor features of shape
      [1, height, width, n_objects, 1].
    """

    if USE_CORRELATION_COST:
        
        d = local_pairwise_distances(query_embedding, prev_frame_embedding,
                                   max_distance=max_distance)
        
    else:
        d = local_pairwise_distances2(query_embedding, prev_frame_embedding,
                                    max_distance=max_distance)
    height,width=prev_frame_embedding.size()[:2]

    if USE_CORRELATION_COST:
        corr_op=SpatialCorrelationSampler(kernel_size=1, patch_size=2*max_distance+1, stride=1,dilation_patch=1,padding=0)
        # New, faster code with cross-correlation via correlation_cost.
      # Due to padding we have to add 1 to the labels.
        tmp_prev_frame_labels=(prev_frame_labels+1).float().permute(2,0,1)
        tmp_prev_frame_labels=paddle.unsqueeze(tmp_prev_frame_labels,0)
        ones_ = paddle.ones_like(tmp_prev_frame_labels)


        offset_labels = corr_op(ones_,tmp_prev_frame_labels)
        bs,_,_,hh,ww=offset_labels.size()
        offset_labels=offset_labels.view(bs,-1,hh,ww)


        offset_labels = paddle.squeeze(offset_labels,0)
        offset_labels = offset_labels.permute(1,2,0)
        offset_labels = paddle.unsqueeze(offset_labels,3)
        offset_labels = paddle.round(offset_labels - 1)
        offset_masks = paddle.eq(
          offset_labels,
          gt_ids.float().unsqueeze(0).unsqueeze(0).unsqueeze(0))


    else:
        if MODEL_UNFOLD:

            labels = prev_frame_labels.float().permute(2, 0, 1).unsqueeze(0)
            padded_labels = F.pad(labels,
                                (2 * max_distance, 2 * max_distance,
                                 2 * max_distance, 2 * max_distance,
                                 ))
            offset_labels = F.unfold(padded_labels, kernel_size=(height, width), stride=(2, 2)).view(height, width, -1, 1)
            offset_masks = paddle.eq(
                offset_labels,
                gt_ids.float().unsqueeze(0).unsqueeze(0).unsqueeze(0))
        else:

            masks = paddle.eq(prev_frame_labels, gt_ids.unsqueeze(0).unsqueeze(0))
            padded_masks = nn.functional.pad(masks,
                            (0,0,max_distance, max_distance,
                             max_distance, max_distance,
                             ))
            offset_masks = []
            for y_start in range(2 * max_distance + 1):
                y_end = y_start + height
                masks_slice = padded_masks[y_start:y_end]
                for x_start in range(2 * max_distance + 1):
                    x_end = x_start + width
                    offset_mask = masks_slice[:, x_start:x_end]
                    offset_masks.append(offset_mask)
            offset_masks = paddle.stack(offset_masks, dim=2)



    d_tiled = d.unsqueeze(-1).repeat((1,1,1,gt_ids.size(0)))
    pad = paddle.ones_like(d_tiled)
    d_masked = paddle.where(offset_masks, d_tiled, pad)
    dists,_ = paddle.min(d_masked, dim=2)
    dists = dists.view(1, height, width, gt_ids.size(0), 1)

    return dists
    
##############################################################


#################
class _res_block(nn.Layer):
    def __init__(self,in_dim,out_dim):
        super(_res_block,self).__init__()
        self.conv1=nn.Conv2D(in_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.relu1=nn.ReLU()
        self.bn1 = SynchronizedBatchNorm2d(out_dim, momentum=cfg.TRAIN_BN_MOM)
        self.conv2=nn.Conv2D(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.bn2 = SynchronizedBatchNorm2d(out_dim, momentum=cfg.TRAIN_BN_MOM)
    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x += res
        return x


####################
class IntSegHead(nn.Layer):
    def __init__(self,in_dim=(cfg.MODEL_SEMANTIC_EMBEDDING_DIM+3),emb_dim=cfg.MODEL_HEAD_EMBEDDING_DIM):
        super(IntSegHead,self).__init__()
        self.conv1 = nn.Conv2D(in_dim,emb_dim,kernel_size=7,stride=1,padding=3)
        self.bn1 = SynchronizedBatchNorm2d(emb_dim, momentum=cfg.TRAIN_BN_MOM)
        self.relu1 = nn.ReLU(True)
        self.res1 = _res_block(emb_dim,emb_dim)
        self.res2 = _res_block(emb_dim,emb_dim)
        self.conv2 = nn.Conv2D(256,emb_dim,kernel_size=3,stride=1,padding=1)
        self.bn2 = SynchronizedBatchNorm2d(emb_dim, momentum=cfg.TRAIN_BN_MOM)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2D(emb_dim,1,1,1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x



class _split_separable_conv2d(nn.Layer):
    def __init__(self,in_dim,out_dim,kernel_size=7):
        super(_split_separable_conv2d,self).__init__()
        self.conv1=nn.Conv2D(in_dim,in_dim,kernel_size=kernel_size,stride=1,padding=int((kernel_size-1)/2),groups=in_dim)
        self.relu1=nn.ReLU(True)
        self.bn1 = SynchronizedBatchNorm2d(in_dim, momentum=cfg.TRAIN_BN_MOM)
        self.conv2=nn.Conv2D(in_dim,out_dim,kernel_size=1,stride=1)
        self.relu2=nn.ReLU(True)
        self.bn2 = SynchronizedBatchNorm2d(out_dim, momentum=cfg.TRAIN_BN_MOM)
        kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')
        kaiming_normal_(self.conv2.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class DynamicSegHead(nn.Layer):
    def __init__(self,in_dim=(cfg.MODEL_SEMANTIC_EMBEDDING_DIM+3),embed_dim=cfg.MODEL_HEAD_EMBEDDING_DIM,kernel_size=1):
        super(DynamicSegHead,self).__init__()
        self.layer1=_split_separable_conv2d(in_dim,embed_dim)
        self.layer2=_split_separable_conv2d(embed_dim,embed_dim)
        self.layer3=_split_separable_conv2d(embed_dim,embed_dim)
        self.layer4=_split_separable_conv2d(embed_dim,embed_dim)
        self.conv=nn.Conv2D(embed_dim,1,1,1)
        kaiming_normal_(self.conv.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv(x)
        return x

##################

    ###############
class IntVOS(nn.Layer):
    def __init__(self,cfg,feature_extracter):
        super(IntVOS,self).__init__()
        self.feature_extracter=feature_extracter ##embedding extractor
        self.feature_extracter.cls_conv=nn.Sequential()
        self.feature_extracter.upsample4=nn.Sequential()
        self.semantic_embedding=None
        self.seperate_conv=nn.Conv2D(cfg.MODEL_ASPP_OUTDIM,cfg.MODEL_ASPP_OUTDIM,kernel_size=3,stride=1,padding=1,groups=cfg.MODEL_ASPP_OUTDIM)
        self.bn1 = SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM)
        self.relu1=nn.ReLU(True)
        self.embedding_conv = nn.Conv2D(cfg.MODEL_ASPP_OUTDIM,cfg.MODEL_SEMANTIC_EMBEDDING_DIM,1,1)
        self.relu2=nn.ReLU(True)
        self.bn2  = SynchronizedBatchNorm2d(cfg.MODEL_SEMANTIC_EMBEDDING_DIM, momentum=cfg.TRAIN_BN_MOM)
        self.semantic_embedding=nn.Sequential(*[self.seperate_conv,self.bn1,self.relu1,self.embedding_conv,self.bn2,self.relu2])

        for m in self.semantic_embedding:
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        
        self.dynamic_seghead = DynamicSegHead() #propagation segm head
        if cfg.MODEL_USEIntSeg:
            self.inter_seghead = IntSegHead(in_dim=cfg.MODEL_SEMANTIC_EMBEDDING_DIM+3)
        else:
            self.inter_seghead = DynamicSegHead(in_dim=cfg.MODEL_SEMANTIC_EMBEDDING_DIM+2) #interaction segm head
        
    def forward(self,x=None,ref_scribble_label=None,previous_frame_mask=None,normalize_nearest_neighbor_distances=True,use_local_map=True,
                seq_names=None,gt_ids=None,k_nearest_neighbors=1,global_map_tmp_dic=None,local_map_dics=None,interaction_num=None,start_annotated_frame=None,
                frame_num=None):

        x=self.extract_feature(x)
        ref_frame_embedding, previous_frame_embedding,current_frame_embedding = paddle.split(x,split_size_or_sections=int(x.size(0)/3), dim=0)

        if global_map_tmp_dic is None:
            dic= self.prop_seghead(ref_frame_embedding,previous_frame_embedding,
                    current_frame_embedding,ref_scribble_label,previous_frame_mask,
                    normalize_nearest_neighbor_distances,use_local_map,
                    seq_names,gt_ids,k_nearest_neighbors,global_map_tmp_dic,local_map_dics,interaction_num,start_annotated_frame,frame_num,self.dynamic_seghead)
            return dic

        else:
            dic, global_map_tmp_dic= self.prop_seghead(ref_frame_embedding,previous_frame_embedding,
                    current_frame_embedding,ref_scribble_label,previous_frame_mask,
                    normalize_nearest_neighbor_distances,use_local_map,
                    seq_names,gt_ids,k_nearest_neighbors,global_map_tmp_dic,local_map_dics,interaction_num,start_annotated_frame,frame_num,self.dynamic_seghead)
            return dic, global_map_tmp_dic


    def extract_feature(self,x):
        x=self.feature_extracter(x)
        x=self.semantic_embedding(x)
        return x
    
    def prop_seghead(self,ref_frame_embedding=None,previous_frame_embedding=None, current_frame_embedding=None,
                ref_scribble_label=None,previous_frame_mask=None,normalize_nearest_neighbor_distances=True,use_local_map=True,
                seq_names=None,gt_ids=None,k_nearest_neighbors=1,global_map_tmp_dic=None,local_map_dics=None,interaction_num=None,start_annotated_frame=None,
                frame_num=None,dynamic_seghead=None):

        """return: feature_embedding,global_match_map,local_match_map,previous_frame_mask"""
    ###############
        global_map_tmp_dic=global_map_tmp_dic
        dic_tmp={}
        bs,c,h,w = current_frame_embedding.size()
        if cfg.TEST_MODE:
            scale_ref_scribble_label=ref_scribble_label.float()
        else:
            scale_ref_scribble_label=paddle.nn.functional.interpolate(ref_scribble_label.float(),size=(h,w),mode='nearest')
        scale_ref_scribble_label = scale_ref_scribble_label.int()
        scale_previous_frame_label=paddle.nn.functional.interpolate(previous_frame_mask.float(),size=(h,w),mode='nearest')
        scale_previous_frame_label=scale_previous_frame_label.int()
        for n in range(bs):
            seq_current_frame_embedding = current_frame_embedding[n]
            seq_ref_frame_embedding = ref_frame_embedding[n]
            seq_prev_frame_embedding = previous_frame_embedding[n]

            seq_ref_frame_embedding = seq_ref_frame_embedding.permute(1,2,0)
            seq_current_frame_embedding = seq_current_frame_embedding.permute(1,2,0)
            seq_ref_scribble_label = scale_ref_scribble_label[n].permute(1,2,0)
            #########Global Map
            nn_features_n, ref_obj_ids = nearest_neighbor_features_per_object(reference_embeddings=seq_ref_frame_embedding, query_embeddings=seq_current_frame_embedding, 
                reference_labels=seq_ref_scribble_label, k_nearest_neighbors=k_nearest_neighbors, gt_ids=gt_ids[n], n_chunks=10)
            if normalize_nearest_neighbor_distances:
                nn_features_n = (paddle.sigmoid(nn_features_n) - 0.5) * 2

            ###
            if global_map_tmp_dic is not None: ###when testing, use global map memory
                if seq_names[n] not in global_map_tmp_dic:
                    global_map_tmp_dic[seq_names[n]]=paddle.ones_like(nn_features_n).repeat(104,1,1,1,1)
                    if  paddle.cuda.is_available():
                        global_map_tmp_dic[seq_names[n]]=global_map_tmp_dic[seq_names[n]].cuda()
                nn_features_n=paddle.where(nn_features_n<=global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0),nn_features_n,global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0))
                
                global_map_tmp_dic[seq_names[n]][frame_num[n]]=nn_features_n.detach()
            
            #########################Local dist map
            seq_prev_frame_embedding = seq_prev_frame_embedding.permute(1,2,0)
            seq_previous_frame_label = scale_previous_frame_label[n].permute(1,2,0)

            if use_local_map:
                prev_frame_nn_features_n = local_previous_frame_nearest_neighbor_features_per_object(prev_frame_embedding=seq_prev_frame_embedding, 
                            query_embedding=seq_current_frame_embedding, prev_frame_labels=seq_previous_frame_label,
                                gt_ids=ref_obj_ids, max_distance=cfg.MODEL_MAX_LOCAL_DISTANCE)
            else:
                prev_frame_nn_features_n,_ = nearest_neighbor_features_per_object(reference_embeddings=seq_prev_frame_embedding, query_embeddings=seq_current_frame_embedding, 
                reference_labels=seq_previous_frame_label, k_nearest_neighbors=k_nearest_neighbors, gt_ids=gt_ids[n], n_chunks=20)
                prev_frame_nn_features_n = (paddle.sigmoid(prev_frame_nn_features_n) - 0.5) * 2

            #############
            if local_map_dics is not None: ##When testing, use local map memory
                local_map_tmp_dic,local_map_dist_dic = local_map_dics
                if seq_names[n] not in local_map_dist_dic:
                    local_map_dist_dic[seq_names[n]] = paddle.zeros(104,9)
                    if paddle.cuda.is_available():
                        local_map_dist_dic[seq_names[n]]=local_map_dist_dic[seq_names[n]].cuda()
                if seq_names[n] not in local_map_tmp_dic:
                    local_map_tmp_dic[seq_names[n]] = paddle.zeros_like(prev_frame_nn_features_n).unsqueeze(0).repeat(104,9,1,1,1,1)
                #    if paddle.cuda.is_available():
                #        local_map_tmp_dic[seq_names[n]] = local_map_tmp_dic[seq_names[n]].cuda()
                local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num-1]=1.0/(abs(frame_num[n]-start_annotated_frame))
                local_map_tmp_dic[seq_names[n]][frame_num[n]][interaction_num-1]=prev_frame_nn_features_n.squeeze(0).detach()
                if interaction_num==1:
                    prev_frame_nn_features_n = local_map_tmp_dic[seq_names[n]][frame_num[n]][interaction_num-1]
                    prev_frame_nn_features_n = prev_frame_nn_features_n.unsqueeze(0)
                else:
                    if local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num-1]>local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num-2]:
                        prev_frame_nn_features_n = local_map_tmp_dic[seq_names[n]][frame_num[n]][interaction_num-1]
                        prev_frame_nn_features_n = prev_frame_nn_features_n.unsqueeze(0)
                    else:
                        prev_frame_nn_features_n = local_map_tmp_dic[seq_names[n]][frame_num[n]][interaction_num-2]
                        prev_frame_nn_features_n = prev_frame_nn_features_n.unsqueeze(0)
                        
                local_map_dics=(local_map_tmp_dic,local_map_dist_dic)
    
            to_cat_previous_frame = (seq_previous_frame_label.float()== ref_obj_ids.float())
    
            to_cat_current_frame_embedding = current_frame_embedding[n].unsqueeze(0).repeat((ref_obj_ids.size(0),1,1,1))
    
            to_cat_nn_feature_n = nn_features_n.squeeze(0).permute(2,3,0,1)
            to_cat_previous_frame = to_cat_previous_frame.unsqueeze(-1).permute(2,3,0,1).float()
            to_cat_prev_frame_nn_feature_n = prev_frame_nn_features_n.squeeze(0).permute(2,3,0,1)
            to_cat = paddle.cat((to_cat_current_frame_embedding,to_cat_nn_feature_n,to_cat_prev_frame_nn_feature_n,to_cat_previous_frame),1)
            pred_=dynamic_seghead(to_cat)
            pred_ = pred_.permute(1,0,2,3)
            dic_tmp[seq_names[n]]=pred_

        if global_map_tmp_dic is None:
            return dic_tmp
        else:
            if local_map_dics is None:
                return dic_tmp,global_map_tmp_dic
            else:
                return dic_tmp,global_map_tmp_dic,local_map_dics

    def int_seghead(self,ref_frame_embedding=None,ref_scribble_label=None,prev_round_label=None,normalize_nearest_neighbor_distances=True,global_map_tmp_dic=None,
                local_map_dics=None,interaction_num=None,seq_names=None,gt_ids=None,k_nearest_neighbors=1,frame_num=None,first_inter=True):
        dic_tmp={}
        bs,c,h,w = ref_frame_embedding.size()
        scale_ref_scribble_label=paddle.nn.functional.interpolate(ref_scribble_label.float(),size=(h,w),mode='nearest')
        scale_ref_scribble_label = scale_ref_scribble_label.int()
        if not first_inter:
            scale_prev_round_label=paddle.nn.functional.interpolate(prev_round_label.float(),size=(h,w),mode='nearest')
            scale_prev_round_label = scale_prev_round_label.int()
        if USE_CORRELATION_COST:
            n_chunks = 20
        else:
            n_chunks = 500
        for n in range(bs):

            gt_id = paddle.arange(0,gt_ids[n]+1)

            gt_id = gt_id.int()
            if paddle.cuda.is_available():
                gt_id = gt_id.cuda()

            seq_ref_frame_embedding = ref_frame_embedding[n]

            ########################Local dist map
            seq_ref_frame_embedding = seq_ref_frame_embedding.permute(1,2,0)
            seq_ref_scribble_label = scale_ref_scribble_label[n].permute(1,2,0)
            nn_features_n = local_previous_frame_nearest_neighbor_features_per_object(prev_frame_embedding=seq_ref_frame_embedding, 
                            query_embedding=seq_ref_frame_embedding, prev_frame_labels=seq_ref_scribble_label,
                                gt_ids=gt_id, max_distance=cfg.MODEL_MAX_LOCAL_DISTANCE)


            #######
            ######################Global map update
            if seq_names[n] not in global_map_tmp_dic:
                global_map_tmp_dic[seq_names[n]]=paddle.ones_like(nn_features_n).repeat(104,1,1,1,1)
            nn_features_n_=paddle.where(nn_features_n<=global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0),nn_features_n,global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0))
            
            ###

            ###
            global_map_tmp_dic[seq_names[n]][frame_num[n]]=nn_features_n_.detach()
            ##################Local map update
            if local_map_dics is not None:
                local_map_tmp_dic,local_map_dist_dic = local_map_dics
                if seq_names[n] not in local_map_dist_dic:
                    local_map_dist_dic[seq_names[n]] = paddle.zeros(104,9)
                    if paddle.cuda.is_available():
                        local_map_dist_dic[seq_names[n]]=local_map_dist_dic[seq_names[n]].cuda()
                if seq_names[n] not in local_map_tmp_dic:
                    local_map_tmp_dic[seq_names[n]] = paddle.ones_like(nn_features_n).unsqueeze(0).repeat(104,9,1,1,1,1)
                local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num-1]=0
                
    
                local_map_dics=(local_map_tmp_dic,local_map_dist_dic)



            ##################
            to_cat_current_frame_embedding = ref_frame_embedding[n].unsqueeze(0).repeat((gt_id.size(0),1,1,1))
            to_cat_nn_feature_n = nn_features_n.squeeze(0).permute(2,3,0,1)
            

            to_cat_scribble_mask_to_cat = (seq_ref_scribble_label.float()== gt_id.float())
            to_cat_scribble_mask_to_cat = to_cat_scribble_mask_to_cat.unsqueeze(-1).permute(2,3,0,1).float()
            if not first_inter:
                seq_prev_round_label = scale_prev_round_label[n].permute(1,2,0)

                to_cat_prev_round_to_cat = (seq_prev_round_label.float()==gt_id.float())
                to_cat_prev_round_to_cat = to_cat_prev_round_to_cat.unsqueeze(-1).permute(2,3,0,1).float()
            else:
                to_cat_prev_round_to_cat = paddle.zeros_like(to_cat_scribble_mask_to_cat)
                to_cat_prev_round_to_cat[0]=1.

            to_cat = paddle.cat((to_cat_current_frame_embedding,to_cat_scribble_mask_to_cat,to_cat_prev_round_to_cat),1)

            pred_ = self.inter_seghead(to_cat)
            pred_ = pred_.permute(1,0,2,3)
            dic_tmp[seq_names[n]]=pred_
        if local_map_dics is None:
            return dic_tmp
        else:
            return dic_tmp,local_map_dics








