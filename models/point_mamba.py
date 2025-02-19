from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from knn_cuda import KNN
from .block import Block
from .build import MODELS

from torch.autograd import Function
from typing import Tuple
import pointops_cuda
import torch.nn.functional as F

def group_by_umbrella(xyz, new_xyz, k=9, cuda=False):
    """
    Group a set of points into umbrella surfaces
    input xyz (3,1024,3)  new_xyz (3,1024,3)

    """
    #print("group_by_umbrella")
    idx = query_knn_point(k, xyz, new_xyz, cuda=cuda)  # 找相邻点 (b,1024,9)
    torch.cuda.empty_cache()
    group_xyz = index_points(xyz, idx)[:, :, 1:]  # [B, N', K-1, 3] #neighborhood (b,1024,8)
    group_xyz_0 = index_points(xyz, idx)[:, :, :1] #(B, N, 1, 3) 4, 1024,1,3)
    group_xyz_0_0=group_xyz_0.repeat(1,1,k-1,1)
    dist=torch.sub(group_xyz,group_xyz_0_0).pow(exponent=2).sum(dim=-1,keepdim=True)
    distance=torch.sqrt(dist) #(B,N,k-1,1)
    torch.cuda.empty_cache()

    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)  # (3,1024,8,3)-(3,1024,1,3) normalization
    temp = xyz2sphere(group_xyz_norm)
    group_phi = temp[..., 2]  # [B, N', K-1] (3, 1024, 8)
    sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1] (3, 1024, 8)

    # [B, N', K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)  # (3, 1024, 8, 1, 3)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)  # all value=0, (3,1024,8,1,3)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)  # (3,1024,8,3,3)

    return umbrella_group_xyz, distance ##(3,1024,8,3,3)

def cal_normal(group_xyz, random_inv=False, is_group=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    :param group_xyz: [B, N, K=3, 3] / [B, N, G, K=3, 3]
    :param random_inv:
    :param return_intersect:
    :param return_const:
    :return: [B, N, 3]
    """
    #print("call_normal") #group_xyz (1,1024,8,3,3)
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [B, N, 3] (3,2048,8,3) s-g-xyz -g-cen(all 0)
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [B, N, 3] (3,2048,8,3) s-g-xyz-roll -g-cen(all 0)

    nor = torch.cross(edge_vec1, edge_vec2, dim=-1) #(3,2048,8,3)#Returns the cross product of vectors in dimension dim of input and other
    unit_nor = nor / torch.norm(nor, dim=-1, keepdim=True)  #(3,2048,8,3) / (3,1024,8,1) = (3,1024,8,3)
    if not is_group:
        pos_mask = (unit_nor[..., 0] > 0).float() * 2. - 1.
    else: #this way
        pos_mask = (unit_nor[..., 0:1, 0] > 0).float() * 2. - 1. #(3,2048,1)# keep x_n positive (mask) all value=1 or -1
    unit_nor = unit_nor * pos_mask.unsqueeze(-1) #(3,2048,8,3)

    # batch-wise !! random inverse normal vector (prob: 0.5)
    if random_inv:   #True this way
        random_mask = torch.randint(0, 2, (group_xyz.size(0), 1, 1)).float() * 2. - 1.  #(3,1,1) eg.-1,1,1 #all value=1 or -1
        random_mask = random_mask.to(unit_nor.device)
        if not is_group:
            unit_nor = unit_nor * random_mask
        else: #this way
            unit_nor = unit_nor * random_mask.unsqueeze(-1)  #(3,2048,8,3)

    return unit_nor #(3,2048,8,3)

def cal_center(group_xyz):
    center = torch.mean(group_xyz, dim=-2)
    return center  # (b,128,4,8,3)

def xyz2sphere(xyz, normalize=True):
    # xyz(b,128,4,8,3) 0 1 2 3,4  #umbrella_xyz=(b,128,4,8,3,3)
    # print("xyz2sphere")
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))  # (b,128,4,8,1)
    rho = torch.clamp(rho, min=0)  # range: [0, inf] #(b,128,4,8,1)
    theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi] #(b,128,4,8,1)
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi] #(b,128,4,8,1)
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)  # (b,128,4,8,3)
    return out

def cal_const(normal, center, is_normalize=True):
    # input: normal=(b,128,4,8,3), center=(b,128,4,8,3)
    const = torch.sum(normal * center, dim=-1, keepdim=True)
    factor = torch.sqrt(torch.Tensor([3])).to(normal.device)
    const = const / factor if is_normalize else const

    return const  # (b,128,4,8,1)

def check_nan_umb(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor
    :param pos: [B, N, G, 1]
    :param center: [B, N, G, 3]
    :param normal: [B, N, G, 3]
    :return:
    """
    # print("check_nan_umb")
    B, N, G, _ = normal.shape  # B,1024,8
    #print("how many nan before", torch.sum(torch.isnan(normal)))
    # t = torch.isnan(normal) #(3,1024,8,3)
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0  # (3,1024,8) #000-fff
    mask_first = torch.argmax((~mask).int(), dim=-1)  # (3,1024,8)-(3,1024)
    b_idx = torch.arange(B).unsqueeze(1).repeat([1, N])  # (3,1024)
    n_idx = torch.arange(N).unsqueeze(0).repeat([B, 1])  # (3,1024)

    normal_first = normal[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])  # (3,1024,8,3)
    normal[mask] = normal_first[mask]
    #print("how many nan after", torch.sum(torch.isnan(normal)))
    center_first = center[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])  # (3,1024,8,3)
    center[mask] = center_first[mask]

    if pos is not None:
        pos_first = pos[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
        pos[mask] = pos_first[mask]
        return normal, center, pos
    return normal, center

class SurfaceAbstractionCD_sa1(nn.Module):

    def __init__(self, npoint, radius, nsample, feat_channel, pos_channel, mlp, group_all,
                 return_normal=True, return_polar=False):
        super(SurfaceAbstractionCD_sa1, self).__init__()
        self.npoint = npoint # int  512
        self.radius = radius # float 0.2
        self.nsample = nsample # int 32
        self.feat_channel=feat_channel # int 10
        self.pos_channel = pos_channel # int 6

        self.return_normal = return_normal # bool
        self.return_polar = return_polar # bool
        #self.cuda = cuda
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.group_all = group_all # bool False


        self.mlp_f0 = nn.Conv2d(self.feat_channel, mlp[0], 1)
        self.mlp_l0 = nn.Conv2d(self.pos_channel, mlp[0], 1)

        self.bn_l0 = nn.BatchNorm2d(mlp[0])
        self.bn_f0 = nn.BatchNorm2d(mlp[0])

        # mlp_l0+mlp_f0 can be considered as the first layer of mlp_convs
        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, center, normal, feature):
        #normal = normal.permute(0, 2, 1)
        #center = center.permute(0, 2, 1)
        #print("Sa!!")
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all: #false
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_normal=self.return_normal,
                                                                       return_polar=self.return_polar)
        else:# this way
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_normal=self.return_normal,
                                                                   return_polar=self.return_polar, cuda=self.cuda)

        new_feature = new_feature.permute(0, 3, 2, 1) #(3,17,32,512)
        #
        # # init layer
        # #print("6+10")
        loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))#6
        feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))#10
        new_feature = loc + feat
        new_feature = F.relu(new_feature)
        #
        for i, conv in enumerate(self.mlp_convs): #64-128
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        new_center = new_center.permute(0, 2, 1)
        new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature

class SurfaceAbstractionCD(nn.Module):
    """
    Surface Abstraction Module
    self.sa1 = SurfaceAbstractionCD(npoint=512, radius=0.2, nsample=32, feat_channel=10,
                                        pos_channel=6, mlp=[64, 64, 128], group_all=False,
                                        return_polar=True)
    """

    def __init__(self, npoint, radius, nsample, feat_channel, pos_channel, mlp, group_all,
                 return_normal=True, return_polar=False):
        super(SurfaceAbstractionCD, self).__init__()
        self.npoint = npoint # int  512
        self.radius = radius # float 0.2
        self.nsample = nsample # int 32
        self.feat_channel=feat_channel # int 10
        self.pos_channel = pos_channel # int 6


        self.return_normal = return_normal # bool
        self.return_polar = return_polar # bool
        #self.cuda = cuda

        self.group_all = group_all # bool False


    def forward(self, center, normal, feature):
        normal = normal.permute(0, 2, 1)
        center = center.permute(0, 2, 1)
        #print("Sa!!")
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all: #false
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_normal=self.return_normal,
                                                                       return_polar=self.return_polar)
        else:# this way
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_normal=self.return_normal,
                                                                   return_polar=self.return_polar, cuda=self.cuda)


        return new_center, new_normal, new_feature

def query_knn_point(k, xyz, new_xyz, cuda=True):
    # print("query_knn_point")
    if cuda:
        if not xyz.is_contiguous():  # False
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():  # False
            new_xyz = new_xyz.contiguous()
        return knnquery(k, xyz, new_xyz)  # 得到idex(b,128,24)
    dist = square_distance(new_xyz, xyz)
    group_idx = dist.sort(descending=False, dim=-1)[1][:, :, :k]
    return group_idx

class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
        if new_xyz is None:
            new_xyz = xyz
        xyz = xyz.contiguous()  # (b,2048,3)
        new_xyz = new_xyz.contiguous()  # (b,128,3)
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, m, _ = new_xyz.size()  # b:3, m:128
        n = xyz.size(1)  # 2048
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()  # (b,128,32) 建了一个空表
        dist2 = torch.cuda.FloatTensor(b, m, nsample).zero_()  # (b,128,32) 建了一个空表
        #m=torch.tensor(m)
        #nsample = torch.tensor(nsample)
        #print(b, n, m, nsample, xyz.shape, new_xyz.shape, idx.shape, dist2.shape)
        pointops_cuda.knnquery_cuda(b, n, m, nsample, xyz, new_xyz, idx, dist2)
        return idx  # (b,128,32)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knnquery = KNNQuery.apply

def index_points(points, idx):  # (128,32,3) (128,4,9)
    # grouping = Grouping()
    # print("index_points")
    points = grouping(points.transpose(1, 2).contiguous(), idx)  # points.transpose(1, 2) (128,32,3) to (128,3,32)

    return points.permute(0, 2, 3, 1).contiguous()  # (128,3,4,9)--->(128,4,9,3)

# From Surface
class Grouping(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """ #(128,3,32) (128,4,9)
         input: features: (b, c, n), idx : (b, m, nsample) containing the indicies of features to group with
         output: (b, c, m, nsample) (b,3,1024,9)-->(128,3,4,9)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        #print("grouping")
        b, c, n = features.size()  # b=128, c=3, n=32
        _, m, nsample = idx.size()  # _=128, m=4, nsample=9
        output = torch.cuda.FloatTensor(b, c, m, nsample)  # (128,3,4,9)
        pointops_cuda.grouping_forward_cuda(b, c, n, m, nsample, features, idx, output)  # add value
        ctx.for_backwards = (idx, n)
        return output  # output: (b, c, m, nsample) (3, 3, 1024, 9)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input: grad_out: (b, c, m, nsample)
        output: (b, c, n), None
        """
        idx, n = ctx.for_backwards
        b, c, m, nsample = grad_out.size()
        grad_features = torch.cuda.FloatTensor(b, c, n).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.grouping_backward_cuda(b, c, n, m, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping = Grouping.apply

def resort_points(points, idx):
    """
    Resort Set of points along G dim points=(3,1024,8,3)  index=(3,1024,8)
    """
    #print("resort_points")
    device = points.device
    B, N, G, _ = points.shape  # B=4, N=1024, G=8, _=3

    view_shape = [B, 1, 1]  # list [B,1,1]
    repeat_shape = [1, N, G]  # list [1,1024,8]
    #torch.arange(B, dtype=torch.long)=(4,)=[0,1,2,3]
    #torch.arange(B, dtype=torch.long).view(view_shape)=(4,1,1)
    #torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)=(4,1024,8)
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) #(4,1024,8)全部填充自然数 构造了假的点云 for points

    view_shape = [1, N, 1] #(1,1204,1)
    repeat_shape = [B, 1, G] #(4,1,8)
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) #(4,1024,8) for index

    new_points = points[b_indices, n_indices, idx, :]
    #print("new_points")
    return new_points

def sample_and_group(npoint, radius, nsample, center, normal, feature, return_normal=True, return_polar=False, cuda=False):
    #print("group!!")
    fps_idx = farthest_point_sample(center, npoint, cuda=cuda)  #(4,512)
    #print("fps_idx",fps_idx.shape)
    torch.cuda.empty_cache()
    # sample center
    new_center = saindex_points(center, fps_idx, is_group=False) #(4,512,3)
    #print("new_center",new_center.shape)
    torch.cuda.empty_cache()
    # sample normal
    new_normal = saindex_points(normal, fps_idx, is_group=False) #(4,512,11)
    torch.cuda.empty_cache()
    #print("new_normal",new_normal.shape)

    # group
    idx = query_ball_point(radius, nsample, center, new_center) #(4,512,24)
    torch.cuda.empty_cache()
    # group normal
    group_normal = saindex_points(normal, idx, is_group=True)  #(4,512,24,11)
    #print("group_normal",group_normal.shape)
    torch.cuda.empty_cache()
    # group center
    group_center = saindex_points(center, idx, is_group=True)  #(4,512,24,3)
    #print("group_center",group_center.shape)
    torch.cuda.empty_cache()
    group_center_norm = group_center - new_center.unsqueeze(2)  #(4,512,24,3)
    #print("group_center_norm",group_center_norm.shape)
    torch.cuda.empty_cache()

    # group polar
    if return_polar:# this way
        group_polar = xyz2sphere(group_center_norm) #(4,512,24,3)
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1) #(4,512,24,6)
        #print("group_center_norm",group_center_norm.shape)
    if feature is not None: #sa2 this way
        group_feature = saindex_points(feature, idx, is_group=True)
        #print("group_feature",group_feature.shape)
        #print("return normal:",return_normal)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1) if return_normal \
            else torch.cat([group_center_norm, group_feature], dim=-1)
        #print("new_feature:",new_feature.shape) #(2, 128, 24, 272)
    else: #sa1 this way
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1) #(4,512,24,17)

    return new_center, new_normal, new_feature

def farthest_point_sample(xyz, npoint, cuda=False):

    if not xyz.is_contiguous():
        xyz = xyz.contiguous()
    return furthestsampling(xyz, npoint)

class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, m):
        """
        input: xyz: (b, n, 3) and n > m, m: int32
        output: idx: (b, m)
        """
        assert xyz.is_contiguous()
        b, n, _ = xyz.size()
        idx = torch.cuda.IntTensor(b, m)
        temp = torch.cuda.FloatTensor(b, n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n, m, xyz, temp, idx)
        return idx

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthestsampling = FurthestSampling.apply

def saindex_points(points, idx, is_group=False):
    #print("index_points")
    if is_group: #this way idx(1,1024,9)
        points = grouping(points.transpose(1, 2).contiguous(), idx) #points.transpose(1, 2) (3,1024,3) to (3,3,1024)
        return points.permute(0, 2, 3, 1).contiguous()  #(3, 3, 1024, 9)
    else:
        points = gathering(points.transpose(1, 2).contiguous(), idx)  #sa this way
        return points.permute(0, 2, 1).contiguous()

class Gathering(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        input: features: (b, c, n), idx : (b, m) tensor
        output: (b, c, m)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        b, c, n = features.size()
        m = idx.size(1)
        output = torch.cuda.FloatTensor(b, c, m)
        pointops_cuda.gathering_forward_cuda(b, c, n, m, features, idx, output)
        ctx.for_backwards = (idx, c, n)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, c, n = ctx.for_backwards
        b, m = idx.size()
        grad_features = torch.cuda.FloatTensor(b, c, n).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_cuda.gathering_backward_cuda(b, c, n, m, grad_out_data, idx, grad_features.data)
        return grad_features, None


gathering = Gathering.apply

def query_ball_point(radius, nsample, xyz, new_xyz):
    if not xyz.is_contiguous():
        xyz = xyz.contiguous()
    if not new_xyz.is_contiguous():
        new_xyz = new_xyz.contiguous()
    return ballquery(radius, nsample, xyz, new_xyz)

class BallQuery(Function):
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, n, _ = xyz.size()
        m = new_xyz.size(1)
        idx = torch.cuda.IntTensor(b, m, nsample).zero_()
        pointops_cuda.ballquery_cuda(b, n, m, radius, nsample, new_xyz, xyz, idx)
        return idx

    def backward(ctx, a=None):
        return None, None, None, None

ballquery = BallQuery.apply


# class Encoder(nn.Module):  ## Embedding module
#     def __init__(self, encoder_channel):
#         super().__init__()
#         self.encoder_channel = encoder_channel
#         self.first_conv = nn.Sequential(
#             nn.Conv1d(3, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, 256, 1)
#         )
#         self.second_conv = nn.Sequential(
#             nn.Conv1d(512, 512, 1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(512, self.encoder_channel, 1)
#         )
#
#     def forward(self, point_groups):
#         '''
#             point_groups : B G N 3
#             -----------------
#             feature_global : B G C
#         '''
#         bs, g, n, _ = point_groups.shape  #bs, g=64, n=32, _=3
#         point_groups = point_groups.reshape(bs * g, n, 3) #(128,32,3) <-(2*64, 32, 3)
#         # encoder
#         feature = self.first_conv(point_groups.transpose(2, 1))  #(128,256,32)  # (128,32,3)->(128,3,32) channel first 3-128-256 increase dim
#         feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (128,256,1)
#         feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # (128,512,32)  <-- 256+256
#         #feature_global.expand(-1, -1, n) --> (128,256,32)
#         feature = self.second_conv(feature)  # (128,384,32)
#         feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
#         return feature_global.reshape(bs, g, self.encoder_channel)

class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel

        # self.first_conv = nn.Sequential(
        #     nn.Conv1d(16, 128, 1),  # 3--.10 #0.2 (b, 512, 32, 16)  0.1=(b, 128, 24, 272)
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     # nn.LeakyReLU(0.01,inplace=True),
        #     nn.Conv1d(128, 256, 1))


        self.second_conv = nn.Sequential(
            nn.Conv1d(546, 546, 1),
            nn.BatchNorm1d(546),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.01,inplace=True),
            nn.Conv1d(546, self.encoder_channel, 1)  # 384
        )

    def forward(self, point_groups):
        # print("Stephanie")  #(b,128,32,10)
        # print(torch.any(torch.isnan(point_groups)));
        assert not torch.any(torch.isnan(point_groups));  #(b,128,24,272)
        bs, g, n, _ = point_groups.shape  # bs=b g=128(center) n=24(k) num_featrues=272
        point_groups = point_groups.reshape(bs * g, n, 273)  # (b*128,32,10)  #3--10 (4*128, 24, 272)
        # encoder   point_groups.transpose(2,1)=(b*128, 10, 32)
        #feature = self.first_conv(point_groups.transpose(2, 1))  # (b*center, 256, 32) # BG 256 n #nan
        # 0.2=(b*512, 256, 32)   0.1=(b*128, 272, 24)
        #print("after first conv")
        point_groups=point_groups.permute(0,2,1) #(bs*128,272,24)
        feature_global = torch.max(point_groups, dim=2, keepdim=True)[0]  #(b*128,272,1)
        # temp=feature_global.expand(-1,-1,n); #(b*center, 256, 32) n=32
        feature = torch.cat([feature_global.expand(-1, -1, n), point_groups],dim=1)  # (b*center,544,24)  # BG 512=256*2 n #返回当前张量在某维扩展更大后的张量
        #(b*128, 544, 24)    #0.1=(b*128, 544, 24)
        feature = self.second_conv(feature)  # (b*center, 384, 24)
        # print("after second conv",feature);
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (b*center, 384)  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)  # (b,64,384) #4 128 384

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


@MODELS.register_module()
class PointMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMamba, self).__init__()
        repsurf_channel = 10
        self.num_k=9
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

        self.surfacemlps = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=11, kernel_size=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU(True),
            nn.Conv2d(in_channels=11, out_channels=11, kernel_size=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU(True),
            nn.Conv2d(in_channels=11, out_channels=11, kernel_size=1, bias=False),
        )

        self.sa1 = SurfaceAbstractionCD_sa1(npoint=512, radius=0.1, nsample=24, feat_channel=11,
                                            pos_channel=6, mlp=[128, 128, 256], group_all=False,
                                            return_polar=True)  # feat_channel=10
        self.sa2 = SurfaceAbstractionCD(npoint=128, radius=0.2, nsample=24, feat_channel=256 + repsurf_channel,
                                        pos_channel=6, mlp=[256, 256, 512], group_all=False,
                                        return_polar=True)  # mlp=[128, 128, 256]

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        torch.autograd.set_detect_anomaly(True)
        center=pts
        # surface construction
        group_xyz, distance = group_by_umbrella(center, center, k=self.num_k,
                                                cuda=self.cuda)  # (b,1024,8,3,3) [B, N, K-1, 3 (points), 3 (coord.)]
        # print("diss")
        # normal
        group_normal = cal_normal(group_xyz, random_inv=True, is_group=True)  # -->(b,1024,8,3)# normal vector
        # coordinate
        group_center = cal_center(group_xyz)  # -->(bs,1024,8,3)
        # polar
        group_polar = xyz2sphere(group_center)  # -->(bs,1024,8,3)
        # print("newfeature")
        group_pos = cal_const(group_normal, group_center)  # (3,2048,8,1)
        # print("group_normal:", group_normal.shape)
        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
        new_feature = torch.cat([group_center, group_polar, group_normal, group_pos, distance], dim=-1)  # N+P+CP+1: 11
        # print("dis")
        # a = torch.isnan(new_feature)
        temp1 = new_feature
        new_feature = temp1.permute(0, 3, 2, 1)  # [B, C, G, N] (4,11,8,1024)
        # -->(bs,10,8,1024)
        # mapping
        new_feature1 = self.surfacemlps(new_feature)  # (4,11,8,1024)
        # (bs,10,8,1024)-->(bs,10,8,1024)
        new_feature2 = torch.sum(new_feature1,
                                 dim=2)  # (4,11,1024)     # no keepdim # this way (3,10,1024) (3,10,1,1024)
        # (bs,10,8,1024)-->(bs,10,1,1024)-->(bs,10,1024)
        normal = new_feature2.permute(0, 2, 1)  # (4,1024,11)
        # (bs,10,1024)---> (bs,1024, 10)
        # print("herehere!")
        # (bs,1024,3),(bs,1024,10)-->center:(bs,512,3), normal:(bs,512,10), feature(bs,512,32,16)
        center, normal, feature = self.sa1(center, normal, None)  # center(4,3,512) normal(4,10,512) feature(4,256,512)
        center, normal, feature = self.sa2(center, normal,
                                           feature)  # center(4,128,3) normal(4,128,11) feature(4,128,24,273)
        #neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(feature)  #(4,128,24,273)--->(4, 128, 384)
        pos = self.pos_embed(center)

        # reordering strategy
        center_x = center[:, :, 0].argsort(dim=-1)[:, :, None] #(2,64) -> (2,64,1) x
        center_y = center[:, :, 1].argsort(dim=-1)[:, :, None] #(2,64) -> (2,64,1) y
        center_z = center[:, :, 2].argsort(dim=-1)[:, :, None] #(2,64) -> (2,64,1) z
        group_input_tokens_x = group_input_tokens.gather(dim=1, index=torch.tile(center_x, (
            1, 1, group_input_tokens.shape[-1])))  #(2,64,384)
        group_input_tokens_y = group_input_tokens.gather(dim=1, index=torch.tile(center_y, (
            1, 1, group_input_tokens.shape[-1])))  #(2,64,384)
        group_input_tokens_z = group_input_tokens.gather(dim=1, index=torch.tile(center_z, (
            1, 1, group_input_tokens.shape[-1])))  #(2,64,384)
        pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))  #(2,64,384)
        pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))  #(2,64,384)
        pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))  #(2,64,384)
        group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z],
                                       dim=1)  #(2,192,384)  192=64*3
        pos = torch.cat([pos_x, pos_y, pos_z], dim=1)

        x = group_input_tokens
        # transformer
        x = self.drop_out(x)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = x[:, :].mean(1)
        ret = self.cls_head_finetune(concat_f)
        return ret


class MaskMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm)

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, config=None):
        super().__init__()
        if hasattr(config, "use_external_dwconv_at_last"):
            self.use_external_dwconv_at_last = config.use_external_dwconv_at_last
        else:
            self.use_external_dwconv_at_last = False
        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=config.drop_path)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        x = self.blocks(x, pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


@MODELS.register_module()
class Point_MAE_Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskMamba(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.MAE_decoder = MambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            config=config,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)


    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1
