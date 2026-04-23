"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.   
---------------------------------------------------------------------------------     
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""   


import os, sys
import math  
import json 
import gc
import numpy as np
from typing import Iterable
from tqdm import tqdm    
from prettytable import PrettyTable
from tidecv import TIDE, datasets

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, MetricLogger_progress, SmoothedValue, dist_utils, plot_sample
from ..logger_module import get_logger
from ..extre_module.ops import Profile
from ..extre_module.utils import TQDM, RANK
from ..misc import MetricLogger, SmoothedValue, dist_utils

CLEAR_MEMORY_STEP = 100
TIME_DEBUG = False
logger = get_logger(__name__)

def _compute_encoder_transformer_grad_percentage(model: torch.nn.Module) -> float:
    """Compute percentage of gradients attributed to encoder transformer only.
    This avoids collecting/printing any other stats for speed.
    """
    total_l1 = 0.0
    enc_l1 = 0.0


    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        val = grad.detach().abs().sum().item()
        total_l1 += val
        if name.startswith('encoder.encoder'):
            enc_l1 += val
    if total_l1 <= 0.0 or not math.isfinite(total_l1):
        return 0.0
    return 100.0 * enc_l1 / total_l1

def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()

    print_freq = kwargs.get('print_freq', 10)
    writer: SummaryWriter = kwargs.get('writer', None)
    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)

    plot_train_batch_freq = kwargs.get('plot_train_batch_freq', 12)
    output_dir = kwargs.get('output_dir', None)
    epoches = kwargs.get('epoches', -1)
    verbose_type = kwargs.get('verbose_type', 'origin')
    header = 'Epoch: {}/{}'.format(epoch, epoches)

    if verbose_type == 'origin':
        metric_logger = MetricLogger(delimiter="  ")
    else:
        metric_logger = MetricLogger_progress(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    pbar = enumerate(
          metric_logger.log_every(data_loader, print_freq, header))

    dt = [
        Profile(device=device),
        Profile(device=device),
        Profile(device=device),
        Profile(device=device),
        Profile(device=device)
    ]

    encoder_grad_percentages = []
    cur_iters = epoch * len(data_loader)

    teacher_model = kwargs.get('teacher_model', None)
    if teacher_model is not None:
        teacher_model.eval()

    for i, (samples, targets) in pbar:
        if i % CLEAR_MEMORY_STEP == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if epoch % plot_train_batch_freq == 0 and i == 0:
            if data_loader.dataset.remap_mscoco_category:
                plot_sample((samples, targets), data_loader.dataset.category2name, output_dir / f"train_batch_{epoch}.png", data_loader.dataset.label2category)
            else:
                plot_sample((samples, targets), data_loader.dataset.category2name, output_dir / f"train_batch_{epoch}.png")
        with dt[0]:
            samples = samples.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        teacher_encoder_output_for_distillation = None
        if teacher_model is not None:
            with torch.no_grad():
                if scaler is not None:
                    with torch.autocast(device_type=str(device), cache_enabled=True):
                        t_out = teacher_model(samples)
                else:
                    t_out = teacher_model(samples)

            teacher_encoder_output_for_distillation = t_out

        if scaler is not None:
            with dt[1]:
                with torch.autocast(device_type=str(device), cache_enabled=True):
                    outputs = model(samples, targets=targets,
                                    teacher_encoder_output=teacher_encoder_output_for_distillation)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                logger.warning(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    new_key = key.replace('module.', '')
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with dt[2]:
                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)
                loss = sum(loss_dict.values())

            with dt[3]:
                scaler.scale(loss).backward()

                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
                        getattr(criterion, 'distill_adaptive_params') and \
                        criterion.distill_adaptive_params.get('enabled', False):
                    pct = _compute_encoder_transformer_grad_percentage(model)
                    encoder_grad_percentages.append(pct)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:
            with dt[1]:
                outputs = model(samples, targets=targets,
                            teacher_encoder_output=teacher_encoder_output_for_distillation)
            with dt[2]:
                loss_dict = criterion(outputs, targets, **metas)
                loss: torch.Tensor = sum(loss_dict.values())
            with dt[3]:
                optimizer.zero_grad()
                loss.backward()

                if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
                        getattr(criterion, 'distill_adaptive_params') and \
                        criterion.distill_adaptive_params.get('enabled', False):
                    pct = _compute_encoder_transformer_grad_percentage(model)
                    encoder_grad_percentages.append(pct)

                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

        with dt[4]:
            if ema is not None:
                ema.update(model)

            if self_lr_scheduler:
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)
            else:
                if lr_warmup_scheduler is not None:
                    lr_warmup_scheduler.step()

            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())

            if not math.isfinite(loss_value):
                logger.warning("Loss is {}, stopping training".format(loss_value))
                logger.info(loss_dict_reduced)
                sys.exit(1)

            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if writer and dist_utils.is_main_process() and global_step % 10 == 0:
                writer.add_scalar('Loss/total', loss_value.item(), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    metric_logger.synchronize_between_processes()
    logger.info(f'Averaged stats:{metric_logger}')
    if TIME_DEBUG:
        time_data = [x.t / len(data_loader) for x in dt]
        logger.debug(f"Data_to_Device:{time_data[0]:.6f}s Inference:{time_data[1]:.6f}s Loss:{time_data[2]:.6f}s Weight_Update:{time_data[3]:.6f}s")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, encoder_grad_percentages


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, test_only=False, output_dir=None, yolo_metrice=False, other_platform_model=None):
    if model is not None:
        model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger_progress(delimiter="  ")
    header = 'Test:'

    iou_types = coco_evaluator.iou_types

    dt = [
        Profile(device=device),
        Profile(device=device)
    ]

    coco_pred_json = []
    for samples, targets in metric_logger.log_every(data_loader, 1, header):
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        if model is not None:
            with dt[0]:
                outputs = model(samples)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

            with dt[1]:
                results = postprocessor(outputs, orig_target_sizes)
        else:
            if 'onnx' in other_platform_model:
                with dt[0]:
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                    labels, boxes, scores = other_platform_model['onnx'].run(
                        output_names=None,
                        input_feed={'images': samples.cpu().detach().numpy(), "orig_target_sizes": orig_target_sizes.cpu().detach().numpy()}
                    )

                    results = []
                    for lab, box, sco in zip(labels, boxes, scores):
                        result = dict(labels=torch.from_numpy(lab), boxes=torch.from_numpy(box), scores=torch.from_numpy(sco))
                        results.append(result)
            elif 'engine' in other_platform_model:
                with dt[0]:
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                    output = other_platform_model['engine']({'images': samples.to(device),
                                                             'orig_target_sizes': orig_target_sizes.to(device)})
                    labels, boxes, scores = output['labels'], output['boxes'], output['scores']

                    results = []
                    for lab, box, sco in zip(labels, boxes, scores):
                        result = dict(labels=lab, boxes=box, scores=sco)
                        results.append(result)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
            coco_pred_json.extend(list(coco_evaluator.coco_eval['bbox'].cocoDt.anns.values()))

    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if test_only:
        if model is not None:
            speed = dict(zip(['inference', 'postprocess'], (x.t / len(data_loader.dataset) * 1e3 for x in dt)))
            logger.info(f'Test On BatchSize:{data_loader.batch_size}')
            logger.info(f"Speed: {speed['inference']:.4f}ms inference, {speed['postprocess']:.4f}ms postprocess per image")
            logger.info(f"FPS(inference+postprocess): {1000 / (speed['inference'] + speed['postprocess']):.2f}")
        else:
            inference_speed = dt[0].t / len(data_loader.dataset) * 1e3
            logger.info(f'Test On BatchSize:{data_loader.batch_size}')
            logger.info(f"Speed: {inference_speed:.4f}ms inference per image"       )
            logger.info(f"FPS(inference): {1000 / inference_speed:.2f}")

    if coco_evaluator is not None:
        logger.info("------------------------ COCO Metrice Start ------------------------")
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        if test_only:
            logger.info(f"Saving coco pred[{output_dir / 'pred.json'}] json...")
            with open(output_dir / 'pred.json', 'w') as f:
                json.dump(coco_pred_json, f)
            logger.info("save success.")

            precisions = coco_evaluator.coco_eval['bbox'].eval['precision']
            cat_ids = coco_evaluator.coco_eval['bbox'].params.cat_ids
            results_per_category = []
            for idx, cat_id in enumerate(cat_ids):
                t = []
                nm = coco_evaluator.coco_gt.cats[cat_id]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = -1
                t.append(f'{nm["name"]}')
                t.append(f'{round(ap, 3)}')

                for iou in [0, 5]:
                    precision = precisions[iou, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = -1
                    t.append(f'{round(ap, 3)}')

                for area in [1, 2, 3]:
                    precision = precisions[:, :, idx, area, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = -1
                    t.append(f'{round(ap, 3)}')
                results_per_category.append(list(t))

            model_metrice_table = PrettyTable()
            model_metrice_table.title = "COCO Metrice"
            model_metrice_table.field_names = ['category', 'AP', 'AP_50', 'AP_75', 'AP_s', 'AP_m', 'AP_l']
            for data in results_per_category:
                model_metrice_table.add_row(list(data))

            numeric_data = [list(data)[1:] for data in results_per_category]

            avg_values = []
            for col_idx in range(len(numeric_data[0])):
                col_values = [float(row[col_idx]) for row in numeric_data if float(row[col_idx]) != -1]
                if col_values:
                    avg_value = sum(col_values) / len(col_values)
                    avg_values.append(round(avg_value, 3))
                else:
                    avg_values.append(-1)

            all_row = ['all'] + avg_values
            model_metrice_table.add_row(all_row)

            print(model_metrice_table)

            try:
                logger.info("------------------------ TIDE Metrice Start ------------------------")
                tide = TIDE()
                tide.evaluate_range(datasets.COCO(data_loader.dataset.ann_file), datasets.COCOResult(output_dir / 'pred.json'))
                tide.summarize()
                tide.plot(out_dir=output_dir / 'tide_result')
            except Exception as e:
                logger.error('TIDE failure... skip message:', e)


    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator


