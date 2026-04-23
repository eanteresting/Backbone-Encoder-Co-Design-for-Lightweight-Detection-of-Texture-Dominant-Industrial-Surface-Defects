"""   
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.  
"""    

import torch.nn as nn   
from ..core import register 

     
__all__ = ['DEIM', ]  

     
@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]
    
    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,     
        decoder: nn.Module,
    ):  
        super().__init__()   
        self.backbone = backbone 
        self.decoder = decoder
        self.encoder = encoder 

    def forward(self, x, targets=None, teacher_encoder_output=None):
        x_backbone = self.backbone(x)  # [S3, S4, S5] features from backbone

        encoder_output = self.encoder(x_backbone)
        # tuple: (fpn_features, student_distill_output) or fpn_features (list) if not training or distillation is off.

        student_distill_output = None
        if self.training and isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            x_fpn_features, student_distill_output = encoder_output
        else:
            x_fpn_features = encoder_output

        # if self.training:
        #     print("[DEIM] student feat is None:", student_distill_output is None)
        #     print("[DEIM] teacher feat is None:", teacher_encoder_output is None)

        x_decoder_out = self.decoder(x_fpn_features, targets)

        if self.training and student_distill_output is not None and teacher_encoder_output is not None:

            def inject_distill(d):
                if isinstance(d, dict):
                    d['student_distill_output'] = student_distill_output
                    d['teacher_encoder_output'] = teacher_encoder_output

            # 主输出
            inject_distill(x_decoder_out)

            # 所有可能的分支统一兜底
            for k, v in x_decoder_out.items():

                # 情况 1：单个 dict（如 enc_outputs 是 dict）
                if isinstance(v, dict):
                    inject_distill(v)

                # 情况 2：list / tuple（如 aux_outputs / dn_outputs）
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, dict):
                            inject_distill(item)

        return x_decoder_out

    def deploy(self, ):
        self.eval()
        for m in self.modules():   
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy() 
        return self   
