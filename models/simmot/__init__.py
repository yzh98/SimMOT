# ------------------------------------------------------------------------
# SimMOT
# Copyright (c) 2026 Yuanzhou Huang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved
# ------------------------------------------------------------------------

from .simmot import SimMOT
from structures.args import Args
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.simmot.id_decoder import IDDecoder
from models.simmot.trajectory_modeling import TrajectoryModeling


def build(config: dict):
    # Generate DETR args:
    detr_args = Args()
    # 1. backbone:
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    # 2. transformer:
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]

    detr_framework = config["DETR_FRAMEWORK"].lower()
    match detr_framework:
        case "deformable_detr":
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
        case _:
            raise NotImplementedError(f"DETR framework {config['DETR_FRAMEWORK']} is not supported.")

    # Build each component:
    _traj_modeling = TrajectoryModeling(
        trajectory_length=config["REL_PE_LENGTH"],
        feat_dim=config["FEATURE_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        keep_recent_ratio=config["KEEP_RECENT_RATIO"],
        num_memories=config["NUM_MEMORIES"],
        use_checkpoint=config["USE_DECODER_CHECKPOINT"]
    )
    _id_decoder = IDDecoder(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
    ) if config["ONLY_DETR"] is False else None

    # Construct SimMOT model:
    simmot_model = SimMOT(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_traj_modeling,
        id_decoder=_id_decoder,
    )

    return simmot_model, detr_criterion
