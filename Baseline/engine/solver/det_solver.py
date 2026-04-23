""" 
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.  
---------------------------------------------------------------------------------
Modified from D-FINE (https://githupper_bound.com/Peterande/D-FINE)  
Copyright (c) 2024 D-FINE authors. All Rights Reserved. 
"""

import time
import json     
import datetime
import copy
import gc
      
import torch   

from ..misc import dist_utils, stats, get_weight_size

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ..logger_module import get_logger

logger = get_logger(__name__)
coco_name_list = ['ap', 'ap50', 'ap75', 'aps', 'apm', 'apl', 'ar', 'ar50', 'ar75', 'ars', 'arm', 'arl']
      
class DetSolver(BaseSolver):   

    def fit(self, cfg_str):   
        self.train()
        args = self.cfg 

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        logger.info("Start training")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            logger.info("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch, lr_scyedule_save_path=self.output_dir)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        logger.info(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                yolo_metrice=self.cfg.yolo_metrice
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][1]
                top1 = test_stats[k][1]
                logger.info(f'best_stat: {best_stat}')

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            self.criterion.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                logger.info(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                plot_train_batch_freq=args.plot_train_batch_freq,
                output_dir=self.output_dir,
                epoches=args.epoches,
                verbose_type=args.verbose_type,
                teacher_model = self.teacher_model,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                else:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = rho
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 2.0), current_weight * 2.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")


            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                yolo_metrice=self.cfg.yolo_metrice
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{coco_name_list[i]}', v, epoch)

                best_stat_tamp = best_stat.copy()
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][1] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][1])
                else:
                    best_stat_tamp[k] = 0
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][1]

                logger.info(f'best_stat: {best_stat}')

                if best_stat['epoch'] == epoch and self.output_dir:
                    logger.info(f"epoch:{best_stat_tamp['epoch']}->{best_stat['epoch']} ap:{best_stat_tamp[k]:.4f}->{best_stat[k]:.4f}")
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        logger.info("save best_stg2.pth success.")
                    else:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')
                        logger.info("save best_stg1.pth success.")

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    logger.info(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        module.deploy()
        _, model_info = stats(self.cfg, module=module)
        logger.info(f"Model Info(fused) {model_info}")
        get_weight_size(module)
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device, True, self.output_dir, self.cfg.yolo_metrice)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return

    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state

    def val_onnx_engine(self, ):


        self.cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 1

        self.eval()
        logger.warning(f"Only support batch_size=1 for validation")
        if self.cfg.path.endswith('onnx'):
            import onnxruntime as ort
            model = ort.InferenceSession(self.cfg.path)
            logger.info(f"Loading Onnx Model: {self.cfg.path}")
            logger.info(f"Using device: {ort.get_device()}")
            model = {'onnx':model}
        else:
            logger.info(f"error")
            return

        test_stats, coco_evaluator = evaluate(None, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device, True, self.output_dir, self.cfg.yolo_metrice, model)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return


