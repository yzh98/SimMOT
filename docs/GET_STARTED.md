# Getting Started

We recommend creating shell scripts under the `scripts/` directory of the workspace (see [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)) to manage the training and inference commands.

## Training

Training is launched via [Accelerate](https://github.com/huggingface/accelerate). The command below starts a single-process training job; you can also place an `accelerate_config.yaml` file under the `scripts/` directory (see [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)) and pass it via `--config_file` to control GPUs, distributed type, and the number of processes:

```yaml
gpu_ids: 0
distributed_type: NO  # MULTI_GPU, NO
num_processes: 1
main_process_port: 10271
```

```bash
# train_dancetrack.sh
# Core configuration (modify as needed):
PRJ=/path/to/simmot
CACHE=/path/to/SimMOT_cache
CONFIG=r50_deformable_detr_simmot_dancetrack.yaml
DETR_PRETRAIN=r50_deformable_detr_coco_dancetrack.pth
EXP_NAME=dancetrack/simmot

# Execution (no modification needed):
accelerate launch --num_processes=1 ${PRJ}/train.py \
  --exp-name ${EXP_NAME} \
  --config-path ${PRJ}/configs/${CONFIG} \
  --data-root ${CACHE}/datasets \
  --detr-pretrain ${CACHE}/checkpoints/${DETR_PRETRAIN} \
  --outputs-dir ${CACHE}/outputs/${EXP_NAME}
```

## Evaluation

```bash
# eval.sh
# Core configuration (modify as needed):
PRJ=/path/to/simmot
CACHE=/path/to/SimMOT_cache
DATA_ROOT=${CACHE}/datasets
DATASET=DanceTrack
SPLIT=val
CKPT_DIR=${CACHE}/outputs/<exp-name>
CKPT_NAME=checkpoint_9.pth

# Execution (no modification needed):
python ${PRJ}/submit_and_evaluate.py \
  --data-root ${DATA_ROOT} \
  --inference-mode evaluate \
  --config-path ${CKPT_DIR}/train/config.yaml \
  --inference-model ${CKPT_DIR}/${CKPT_NAME} \
  --outputs-dir ${CKPT_DIR} \
  --inference-dataset ${DATASET} \
  --inference-split ${SPLIT}
```

## Submission

```bash
# submit.sh
# Core configuration (modify as needed):
PRJ=/path/to/simmot
CACHE=/path/to/SimMOT_cache
DATA_ROOT=${CACHE}/datasets
DATASET=DanceTrack
SPLIT=test
CKPT_DIR=${CACHE}/outputs/<exp-name>
CKPT_NAME=checkpoint_9.pth

# Execution (no modification needed):
python ${PRJ}/submit_and_evaluate.py \
  --data-root ${DATA_ROOT} \
  --inference-mode submit \
  --config-path ${CKPT_DIR}/train/config.yaml \
  --inference-model ${CKPT_DIR}/${CKPT_NAME} \
  --outputs-dir ${CKPT_DIR} \
  --inference-dataset ${DATASET} \
  --inference-split ${SPLIT}
```