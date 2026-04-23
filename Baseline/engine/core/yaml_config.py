""" 
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)  
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""  

import torch   
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import re    
import copy
  
from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict
from engine.deim.dinov2_teacher import DINOv2TeacherModel
from engine.deim.dinov3_teacher import DINOv3TeacherModel
from ..misc import dist_utils # Assuming remove_module_prefix is there, or define it from _solver.py as needed.


from ..logger_module import get_logger    

logger = get_logger(__name__)

from collections import defaultdict
def debug_optimizer_params(param_groups):
    """ 
    Check for duplicate parameters in optimizer parameter groups

    Args:
        param_groups: Parameter groups list passed to optimizer
    """
    print("=== Optimizer Parameter Duplicate Check ===")

    # Record which groups each parameter appears in
    param_to_groups = defaultdict(list)

    # Iterate through all parameter groups
    for group_idx, group in enumerate(param_groups):
        print(f"\nParameter Group {group_idx}:")

        # Get parameter list
        if isinstance(group, dict) and 'params' in group:
            params = group['params']
            print(f"  - Parameter Count: {len(params)}")
            print(f"  - Other Config: {dict((k, v) for k, v in group.items() if k != 'params')}")
        else:
            params = group
            print(f"  - Parameter Count: {len(params)}")

        # Check each parameter
        for param_idx, param in enumerate(params):
            param_id = id(param)
            param_to_groups[param_id].append((group_idx, param_idx))

            # Print parameter info
            if hasattr(param, 'shape'):
                print(f"    Parameter {param_idx}: shape={param.shape}, id={param_id}")
            else:
                print(f"    Parameter {param_idx}: type={type(param)}, id={param_id}")

    # Check for duplicate parameters
    print("\n=== Duplicate Parameter Check Results ===")
    duplicates_found = False

    for param_id, group_locations in param_to_groups.items():
        if len(group_locations) > 1:
            duplicates_found = True
            print(f"\n❌ Duplicate parameter found (id: {param_id}):")
            print(f"   Appears at: {group_locations}")

            # Try to get more info about the parameter
            for group_idx, param_idx in group_locations:
                if isinstance(param_groups[group_idx], dict):
                    param = param_groups[group_idx]['params'][param_idx]
                else:
                    param = param_groups[group_idx][param_idx]

                if hasattr(param, 'shape'):
                    print(f"   - Group{group_idx}[{param_idx}]: shape={param.shape}")
                if hasattr(param, '_get_name'):
                    print(f"   - Group{group_idx}[{param_idx}]: name={param._get_name()}")

    if not duplicates_found:
        print("✅ No duplicate parameters found")

    return duplicates_found

def check_model_params_with_names(model):
    """
    Check model parameter grouping and return parameter ID to name mapping
    """
    print("=== Model Parameter Analysis ===")

    all_params = list(model.parameters())
    named_params = dict(model.named_parameters())

    print(f"Total Model Parameters: {len(all_params)}")
    print(f"Named Parameters: {len(named_params)}")

    # Check parameter names and corresponding parameter objects
    param_id_to_name = {}
    for name, param in named_params.items():
        param_id_to_name[id(param)] = name

    print("\nParameter Name Mapping:")
    for param_id, name in param_id_to_name.items():
        print(f"  {name}: id={param_id}")

    return param_id_to_name

# Add this debugging code before creating the optimizer:
def debug_before_optimizer_creation(model, param_groups_or_params):
    """
    Debug before optimizer creation
    """
    print("=" * 50)
    print("Starting Optimizer Parameter Debug")
    print("=" * 50)

    # Analyze model parameters
    param_id_to_name = check_model_params_with_names(model)

    # Check parameters passed to optimizer
    if isinstance(param_groups_or_params, (list, tuple)):
        print(f"\nParameter groups passed to optimizer: {len(param_groups_or_params)}")
        debug_optimizer_params(param_groups_or_params)
    else:
        print("\nSingle parameter list passed to optimizer")
        debug_optimizer_params([param_groups_or_params])

    print("=" * 50)
    print("Debug Complete")
    print("=" * 50)

class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return super().model

    @property
    def teacher_model(self, ) -> torch.nn.Module:
        if self._teacher_model is None and 'teacher_model' in self.yaml_cfg:
            teacher_model_cfg = self.yaml_cfg['teacher_model']

            model_type = teacher_model_cfg.pop('type', None)
            # more VFMs will be added.
            if model_type == "DINOv3TeacherModel":
                try:
                    self._teacher_model = DINOv3TeacherModel(**teacher_model_cfg)
                    teacher_model_cfg['type'] = model_type
                    if dist_utils.is_main_process():
                        print("Successfully loaded and configured DINOv3 Teacher Model.")
                except Exception as e:
                    print(f"Error creating DINOv3TeacherModel: {e}")
                    teacher_model_cfg['type'] = model_type
                    raise
            elif model_type == "DINOv2TeacherModel":
                try:
                    self._teacher_model = DINOv2TeacherModel(**teacher_model_cfg)
                    if dist_utils.is_main_process():
                        print("Successfully loaded and configured DINOv2 Teacher Model.")
                except Exception as e:
                    print(f"Error creating DINOv2TeacherModel: {e}")
                    teacher_model_cfg['type'] = model_type
                    raise
            else:
                print(
                    f"Configured teacher_model type '{model_type}' does not match expected 'DINOv3TeacherModel'.")
                raise ValueError("Mismatch in teacher model type configuration.")
        return super().teacher_model

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self, ) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)
        return super().criterion

    @property
    def optimizer(self, ) -> optim.Optimizer:
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            # debug_before_optimizer_creation(self.model, params)
            self._optimizer = create('optimizer', self.global_cfg, params=params)
        return super().optimizer

    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create('lr_scheduler', self.global_cfg, optimizer=self.optimizer)
            logger.info(f'Initial lr: {self._lr_scheduler.get_last_lr()}')
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg :
            self._lr_warmup_scheduler = create('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler)
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return super().train_dataloader

    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return super().val_dataloader

    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            self._ema = create('ema', self.global_cfg, model=self.model)
        return super().ema

    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            self._scaler = create('scaler', self.global_cfg)
        return super().scaler

    @property
    def evaluator(self, ):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':
                from ..data import get_coco_api_from_dataset
                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                self._evaluator = create('evaluator', self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return super().evaluator

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert 'type' in cfg, ''
        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters()

        assert isinstance(cfg['params'], list), ''

        param_groups = []
        visited = []
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
            # print(pattern, params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
            visited.extend(list(params.keys()))
            # print(params.keys())

        # from collections import Counter
        # counter = Counter(visited)
        # duplicates = [item for item, count in counter.items() if count > 1]
        # print(duplicates)

        # assert len(visited) == len(names), ''

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided.
        """
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'

        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            from ..misc import dist_utils
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if 'total_batch_size' in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop('total_batch_size')
        logger.info(f'building {name} with batch_size={bs}...')
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)
        return loader

