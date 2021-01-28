from typing import Any, Dict, Sequence, Tuple, Union, cast
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
import time
import os

from collections import OrderedDict
import torch
from torch import nn

from fvcore.common.checkpoint import Checkpointer
from fvcore.common.config import CfgNode as _CfgNode
import determined as det
from determined.pytorch import DataLoader, PyTorchTrial, reset_parameters, LRScheduler

import detectron2
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.config import get_cfg
from detectron2_files.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import build_lr_scheduler
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
    

import detectron2.utils.comm as comm
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class DetectronTrial(PyTorchTrial):
    def __init__(self, context: det.TrialContext) -> None:
        self.context = context

        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.data_downloaded = False
        self.cfg = get_cfg()
        
        self.cfg.merge_from_file(self.context.get_hparam('model_yaml'))
        self.cfg.SOLVER.IMS_PER_BATCH = self.context.get_per_slot_batch_size()

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        step_mode = LRScheduler.StepMode.STEP_EVERY_BATCH
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return LRScheduler(scheduler, step_mode=step_mode)

    def build_training_data_loader(self) -> DataLoader:
        dataset = build_detection_train_loader(self.cfg, batch_size=self.context.get_per_slot_batch_size(), context=self.context, num_workers=None)
        
        return DataLoader(dataset, batch_sampler=AspectRatioGroupedDatasetBatchSamp(dataset, self.context.get_per_slot_batch_size()), collate_fn=PadSequence(), num_workers=4)

    def build_validation_data_loader(self) -> DataLoader:
        dataset = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0],batch_size=self.context.get_per_slot_batch_size(), num_workers=None)
        
        return DataLoader(dataset, batch_sampler=AspectRatioGroupedDatasetBatchSamp(dataset, self.context.get_per_slot_batch_size()), collate_fn=PadSequence())

    def build_model(self) -> nn.Module:
        model = build_model(self.cfg)
        fi = self.context.get_hparam('data_loc')
        ch = DetectionCheckpointer(model)
        ch.resume_or_load(path = fi, resume=False)
        return ch.model

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:  # type: ignore
        optim = build_optimizer(self.cfg, model)
        return optim

    def train_batch(self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int):
        loss_dict = model(batch)
        losses = sum(loss_dict.values())
        loss_dict['loss'] = losses

        return loss_dict

    def evaluate_full_dataset(self,data_loader, model):
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.

        results = OrderedDict()
        for dataset_name in self.cfg.DATASETS.TEST:
            evaluator = get_evaluator(
                self.cfg, dataset_name, os.path.join(self.cfg.OUTPUT_DIR, "inference", dataset_name)
            )

            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
        return results

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator= COCOEvaluator(dataset_name, cfg, False)
        evaluator_list.append(evaluator)


    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        print ('evaluator_list: ',evaluator_list[0])
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)