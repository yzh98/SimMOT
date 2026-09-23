# Copyright (c) 2026 Yuanzhou Huang. All Rights Reserved.

# Y. Huang: 说明
# DanceTrack视频数量多，单个视频的序列较长，在GPU资源有限的情况下训练困难
# 本模块基于DanceTrack原数据集进行采样，将数据集进行缩减
#  缩减方法包括两个部分：(1) 减少视频数量；(2) 减少视频长度
#  1、减少视频数量
#       DanceTrack数据集共包含40个训练集和25个验证集。
#       根据对视频明暗环境、目标大小、形态变化等因素分类，我们从两个数据集中分别挑选10个视频用于训练和验证
# 2、减少视频长度
#       (1) 仅截取视频的前一半，将视频进行减半
#       (2) 对视频隔一帧采样一帧，将视频长度减半

# PS: 本模块在创建数据集的过程中是创建的软连接的方式创建数据集，并不会消耗过多存储空间，且不会影响原数据集结构

import os
from os import path as osp
from pathlib import Path
import argparse
import configparser
import warnings

def mkdir(pth):
    if isinstance(pth, Path):
        if not pth.exists():
            pth.mkdir(parents=True)
    if isinstance(pth, str):
        if not osp.exists(pth):
            os.mkdir(pth)

def symlink(src, dst):
    src = src if isinstance(src, str) else str(src)
    dst = dst if isinstance(dst, str) else str(dst)
    try:
        os.symlink(str(src), str(dst))
    except FileExistsError:
        pass

def create_symlinks(source_folder: Path, target_folder: Path):
    pad = len(str(source_folder))
    for pth in source_folder.rglob('*'):
        if os.path.isfile(str(pth)):
            tgt_pth = os.path.join(str(target_folder), str(pth)[pad+1:])
            root, filename = os.path.split(tgt_pth)
            if filename[0] == '.':  # skip ".DS_Store"
                continue
            root = Path(root)
            if not root.exists():
                root.mkdir(parents=True)
            symlink(pth, tgt_pth)

def name2int(name: str, ext: str=".jpg"):
    """
        name: e.g., "00000045.jpg"
    """
    return int(name[:-len(ext)])

def int2name(idx: int, pad: int=8, ext: str=".jpg"):
    return f"{idx:0>{pad}}{ext}"

if __name__ == '__main__':
    # GET ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                        help='train, train1, train2, val (No test datasets)')
    parser.add_argument('--data_root', type=str, default='/root/datasets/DanceTrack', 
                        help='the root of DanceTrack')
    parser.add_argument('--numbers', type=str, default='6,20,23,29,49,68,74,83,86,96',
                        help='the video number in split of the DanceTrack.')
    # train1 (6,20,23,29,49); train2 (68,74,83,86,96); train (6,20,23,29,49,68,74,83,86,96)
    # val (5,7,19,25,26,34,43,65,77,94)
    parser.add_argument('--reduce_length', type=str, default='pre-half',
                        help='no-use, pre-half, interval (NOTE: we do not reduce the video length if len(vid) < 800)')
    parser.add_argument('--target_directory', type=str, default='/root/cache/SimMOT_cache/datasets/ablate/DanceTrack',
                        help='the new dancetrack directory path')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.optionxform = str

    # CHECK CONFIGS
    assert args.split in ['train', 'train1', 'train2', 'val'], f'invalid config settings ["--split", {args.split}]'
    split = 'train' if args.split in ['train', 'train1', 'train2'] else 'val'
    assert args.reduce_length in ['no-use', 'pre-half', 'interval'], f'invalid config settings ["--reduce_length", {args.reduce_length}]'

    # SET PATH
    data_root = Path(args.data_root)
    tgt_root = Path(args.target_directory)
    assert data_root.exists(), f"path {data_root} is not exists."
    mkdir(tgt_root)

    # CHECK SPLIT AND VIDEO NUMBERS
    split_path = data_root / args.split
    assert split_path.exists(), f"path {split_path} is not exists."
    vid_names = [f'dancetrack{n:0>4}' for n in args.numbers.split(',')]
    split_vid_names = os.listdir(str(split_path))
    available_vid_names = []
    not_exist_vid_names = []
    for name in vid_names:
        if name in split_vid_names:
            available_vid_names.append(name)
        else:
            not_exist_vid_names.append(name)
    if len(not_exist_vid_names) > 0:
        print(f'Warning: The below videos not exist in current {args.split} dataset:\n')
        print("\n".join(not_exist_vid_names))

    # PROCESS VIDEO
    for name in available_vid_names:
        vid_root = split_path / name
        vid_tgt = tgt_root / split / name
        mkdir(vid_tgt / 'img1')
        mkdir(vid_tgt / 'gt')
        # video length -- no reduce 
        if args.reduce_length == 'no-use':
            create_symlinks(vid_root, vid_tgt)
        else:  # reduce half length of the video
            image_names = sorted([f for f in os.listdir(str(vid_root / 'img1')) if f[-4:] == '.jpg'])  # skip .DS_Store
            if args.reduce_length == 'pre-half':
                # get image names
                seq_len = len(image_names) // 2
                if seq_len <= 400:
                    create_symlinks(vid_root, vid_tgt)
                    continue
                image_names = image_names[:seq_len]
                # for images
                for im in image_names:
                    im_src = vid_root / 'img1' / im
                    im_dst = vid_tgt / 'img1' / im
                    symlink(im_src, im_dst)
                # for gt.txt
                with open(osp.join(str(vid_root / 'gt'), 'gt.txt'), 'r') as f:
                    gt_src = f.readlines()
                with open(osp.join(str(vid_tgt / 'gt'), 'gt.txt'), 'w') as f:
                    for line in gt_src:
                        frm_name = f'{line.split(",")[0]:0>8}.jpg'
                        if frm_name in image_names:
                            f.write(line)
                # for seqinfo.ini
                config.read(str(vid_root / 'seqinfo.ini'))
                config.set('Sequence', 'seqLength', str(len(image_names)))
                with open(str(vid_tgt / 'seqinfo.ini'), 'w') as configfile:
                    config.write(configfile)
            else:
                interval = 3
                # get image names
                image_names = image_names[::3]
                image_map = {int(image_names[i][:8]):i+1 for i in range(len(image_names))}
                config.read(str(vid_root / 'seqinfo.ini'))
                ext=config.get('Sequence', 'imExt')
                # for images
                for im in image_names:
                    im_src = vid_root / 'img1' / im
                    im_dst = vid_tgt / 'img1' / int2name(image_map[name2int(im)], pad=8, ext=ext)
                    symlink(im_src, im_dst)
                # for gt.txt
                with open(osp.join(str(vid_root / 'gt'), 'gt.txt'), 'r') as f:
                    gt_src = f.readlines()
                with open(osp.join(str(vid_tgt / 'gt'), 'gt.txt'), 'w') as f:
                    for line in gt_src:
                        frm_info = line.split(",")
                        if int(frm_info[0]) in image_map:
                            frm_info[0] = str(image_map[int(frm_info[0])])
                            new_line = ",".join(frm_info)
                            f.write(new_line)
                # for seqinfo.ini
                config.set('Sequence', 'seqLength', str(len(image_names)))
                with open(str(vid_tgt / 'seqinfo.ini'), 'w') as configfile:
                    config.write(configfile)
                pass
            
    # WRITE <split>_seqmap.txt
    with open(str((tgt_root / f'{split}_seqmap.txt')), 'a+') as f:
        f.seek(0)
        if len(f.readlines()) == 0:
            f.write('name\n')
        else:
            f.write('\n')
        f.write('\n'.join(available_vid_names))
