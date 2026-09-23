# Project Structure

To facilitate code and data management, SimMOT separates data files from the codebase. All training data, checkpoints, outputs, and execution scripts should be stored in a separate workspace folder.

The location and name of this workspace folder are fully customizable — it can be placed inside the code root directory, alongside the code directory, or in any other location. For example, if the code is stored in `projects/SimMOT`, the workspace folder can be placed in `cache/SimMOT_cache`:

```
projects/
└── SimMOT/                    # Code repository
    ├── configs/
    ├── models/
    ├── data/
    ├── train.py
    └── ...

cache/
└── SimMOT_cache/              # Workspace folder (customizable location and name)
    ├── datasets/              # Datasets (DATA_ROOT)
    ├── checkpoints/           # Pretrained weights
    ├── outputs/               # Training outputs and inference results
    ├── scripts/               # Experiment scripts
    └── debug/                 # Debug outputs (small-scale testing)
```

## Recommended Directories

```
SimMOT_cache/
├── datasets/                   # Datasets (DATA_ROOT)
│   ├── DanceTrack/             # DanceTrack dataset + seqmap files
│   ├── SportsMOT/              # SportsMOT dataset
│   └── BFT/                    # BFT dataset
├── checkpoints/                # Pretrained weights (place manually)
│   └── r50_deformable_detr_coco_*.pth
├── scripts/                    # Experiment scripts
│   ├── train.sh                # training script
│   ├── eval.sh                 # Validation evaluation script (evaluate mode)
│   ├── submit.sh               # Test submission script (submit mode)
│   └── accelerate_config.yaml  # Accelerate distributed config
├── outputs/                    # Experiment outputs (training artifacts + inference results)
└── debug/                      # Debug outputs (small-scale testing)
```

## Training Output Structure

Each experiment's output directory follows this structure:

```
${OUTPUTS_DIR}/
├── checkpoint_{epoch}.pth       # Model weights (saved every SAVE_CHECKPOINT_PER_EPOCH epochs)
└── train/
    ├── config.yaml              # Merged full configuration
    ├── log.txt                  # Training log
    ├── train_summary.txt        # Per-epoch metric summary
    ├── eval_res.txt             # Evaluation results during training (if INFERENCE_DATASET is configured)
    └── eval_during_train/       # Per-epoch tracker results during training
        └── epoch_{N}/tracker/
```

## Inference Output Structure

```
${OUTPUTS_DIR}/${MODE}/${GROUP}/${DATASET}/${SPLIT}/${MODEL_NAME}/
├── tracker/
│   ├── ${sequence_name}.txt     # Tracking results (frame,id,x,y,w,h,1,-1,-1,-1)
│   └── pedestrian_summary.txt   # TrackEval metric summary (evaluate mode only)
└── ... (logs, etc.)
```

`${MODEL_NAME}` is the checkpoint filename without the `.pth` suffix (e.g., `checkpoint_8`).
