"""
INSTR network implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
import pytorch_lightning as pl

from model.transformer_utils.transformer import Transformer
from model.transformer_utils.position_encoding import PositionEmbeddingSine
from model.base_modules import Conv2dNormActiv
from model.transformer_utils.segmentation import FPN
from utils.tensorboard_utils import _convert_instanceseg_to_grid, _convert_instanceseg_to_map, _convert_disp
from model.axial_resnet import conv1x1, AxialBlock
from model.subpixel_corr import SubpixelCorrelationWrapper
from utils.confmat import ConfusionMatrix
from model.loss import bipartite_matching_segmentation_loss
from utils.utils import create_optimizer, create_scheduler, pekdict, load_config


class INSTR(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()

        self.config = cfg

        # settings
        self.num_queries = cfg.MODEL.get("NUM_QUERIES", 15)
        self.aux_decoder_loss = cfg.MODEL.get("AUX_DECODER_LOSS", True)
        self.with_disp = cfg.MODEL.get("WITH_DISP", True)
        self.query_proc = cfg.MODEL.get("QUERY_PROC", "expanded")

        try:
            res50 = resnet50(pretrained=True)
            print(f"Loaded pre-trained ResNet50")
        except Exception as e:
            res50 = resnet50(pretrained=False)
            print(f"Failed to load pre-trained ResNet50 weights: {e}")

        self.layer1 = nn.Sequential(res50.conv1, res50.bn1, res50.relu, res50.maxpool, res50.layer1)
        self.layer2 = res50.layer2

        # axial attention for layer 3 and 4
        self.groups = 8
        self.base_width = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        if cfg.MODEL.get("AXIAL_ATTENTION", True):
            self.layer3 = self._make_axatt_layer(AxialBlock, 512, 512, 6, stride=2, kernel_size_height=60,
                                                 kernel_size_width=80, dilate=False)
            self.layer4 = self._make_axatt_layer(AxialBlock, 1024, 1024, 3, stride=2, kernel_size_height=30,
                                                 kernel_size_width=40, dilate=False)
        else:
            self.layer3 = res50.layer3
            self.layer4 = res50.layer4

        # reduction conv
        self.backbone_reduction = nn.Conv2d(2048, 256, 1)

        # transformer
        self.hs_dim = 1 if not self.aux_decoder_loss else 6
        self.transformer = Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                                       return_intermediate_dec=self.aux_decoder_loss, query_proc=self.query_proc,
                                       h=15, w=20)

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
        self.query_embed = nn.Embedding(self.num_queries, 256)

        if self.query_proc == "expanded":
            fpn_dim = 256
        elif self.query_proc == "att":
            fpn_dim = 8
        elif self.query_proc == "attcat_tfenc" or self.query_proc == "attcat_bb":
            fpn_dim = 264
        else:
            raise NotImplementedError

        d2, d1 = 512, 256
        if self.with_disp:
            d1 += 64
            d2 += 32
            self.corr_layer1 = SubpixelCorrelationWrapper(layer='1', mode="bilinear")
            self.corr_layer2 = SubpixelCorrelationWrapper(layer='2', mode="bilinear")
            self.corr_reduction = Conv2dNormActiv(512 + 32, 512, k_size=1, padding=0, norm=nn.BatchNorm2d,
                                                  activation=nn.ReLU)
            self.disp_decoder = FPN(dim=256, fpn_dims=[1024, d2, d1], context_dim=256)

        # query decoder
        self.query_decoder = FPN(dim=fpn_dim, fpn_dims=[1024, d2, d1], context_dim=256)

        # metric
        self.confmat = ConfusionMatrix()

        # loss params
        self.power = cfg.LOSS.get("POWER", 0.2)
        self.pos_weight = cfg.LOSS.get("POS_WEIGHT", 1.)
        self.neg_weight = cfg.LOSS.get("NEG_WEIGHT", 1.)

        print(f"Initialized INSTR:\n{self}")

    def adapt_to_new_intrinsics(self, f_old=541.14, b_old=0.065, f_new=541.14, b_new=0.065):
        """
        Calculates new sampling strategies for the subpixel correlation layers given new focal length and / or baseline
        during inference. See Eq. 2 and 3 in the paper.
        Assumes fixed z_{min} and output_stride.
        Args:
            f_old (float): old focal length
            b_old (float): old baseline in meters
            f_new (float): new focal length
            b_new (float): new baseline in meters
        """
        # assume z_{min} and output_stride stay fixed, we can directly relate old intrinsics to the displacement ratio
        # f_old * b_old .......... 64/160
        # f_new * b_new .......... ??/160
        # we only need to change d_max, since the rest is calculated internally given a fixed c_c
        d_max_new = (self.corr_layer1.corr.d_max * f_new * b_new) / (f_old * b_old)
        self.corr_layer1.corr.calculate_grid(d_max=d_max_new)

        d_max_new = (self.corr_layer2.corr.d_max * f_new * b_new) / (f_old * b_old)
        self.corr_layer2.corr.calculate_grid(d_max=d_max_new)

    def _make_axatt_layer(self, block, inplanes, planes, blocks, kernel_size_height=56, kernel_size_width=56,
                          stride=1, dilate=False):
        # Initializes axial layers with default parameters.
        # See https://github.com/csrhddlam/axial-deeplab for further details
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size_height=kernel_size_height,
                            kernel_size_width=kernel_size_width))
        inplanes = planes * block.expansion
        if stride != 1:
            kernel_size_height = kernel_size_height // 2
            kernel_size_width = kernel_size_width // 2

        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size_height=kernel_size_height,
                                kernel_size_width=kernel_size_width))

        return nn.Sequential(*layers)

    def forward(self, data):
        """
        Forward pass.
        Args:
            data (dict): Holds left (and optionally right) tensor(s) of shape (b, 3, h, w).
                'color_0': key for left image
                'color_1': key for right image (optional)

        Returns:
            dict: results
                'predictions_{i}': prediction tensor of shape (b, num_queries, h, w) containing prediction logits
                'disp_pred': disparity prediction of shape (b, 1, h, w) containing pixel-wise disparity values
                'queries': object queries
        """
        left = data['color_0']

        # encoder forward
        l1_left = self.layer1(left)
        l2_left = self.layer2(l1_left)

        if self.with_disp:
            l1_right = self.layer1(data['color_1'])
            l2_right = self.layer2(l1_right)

            corr1 = self.corr_layer1(l1_left, l1_right)
            corr2 = self.corr_layer2(l2_left, l2_right)
            l1_cat = torch.cat((l1_left, corr1), dim=1)
            l2_cat = torch.cat((l2_left, corr2), dim=1)
            reduced = self.corr_reduction(l2_cat)
        else:
            reduced = l2_left
            l1_cat = l1_left
            l2_cat = l2_left

        # further encode
        l3 = self.layer3(reduced)
        l4 = self.layer4(l3)

        l4_reduced = self.backbone_reduction(l4)
        b = l4_reduced.shape[0]

        # transformer + instanceseg prediction
        mask = torch.zeros(b, 15, 20, dtype=torch.bool, device=l4_reduced.device)
        pos_encoding = self.pos_embed(device=l4_reduced.device, mask=mask).to(l4_reduced.dtype)
        inst_dec, hs_viz, encoded_transformer = self.transformer(l4_reduced, mask, self.query_embed.weight,
                                                                 pos_encoding)
        encoded_transformer = encoded_transformer.permute(1, 2, 0).view(b, 256, 15, 20)

        # see test_viewing() in tests.test_res50_transformer for details and verification of implementation
        inst_pred = self.query_decoder(inst_dec, [l3, l2_cat, l1_cat])
        inst_pred = inst_pred.view(b, self.hs_dim, self.num_queries, l1_left.shape[-2], l1_left.shape[-1])

        rets = pekdict()
        # make sure that the final decoder output is always at pos 0 for loss and tb
        for i, hsd in enumerate(range(self.hs_dim - 1, -1, -1)):
            pred = F.interpolate(inst_pred[:, hsd, :, :, :], (480, 640), mode='bilinear', align_corners=True)
            rets.add(f'predictions_{i}', pred, tb=_convert_instanceseg_to_grid)
            rets.add(f'predictions_map_{i}', pred.clone().detach(), tb=_convert_instanceseg_to_map)

        if self.with_disp:
            disp_pred = self.disp_decoder(encoded_transformer, [l3, l2_cat, l1_cat])
            disp_pred = F.interpolate(disp_pred, size=(480, 640), mode='bilinear', align_corners=False)
            rets.add('disp_pred', disp_pred, tb=_convert_disp)

        return rets

    def configure_optimizers(self):
        optimizer = create_optimizer(cfg=self.config, net=self)
        scheduler = create_scheduler(cfg=self.config, optimizer=optimizer)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        y_hat = self.forward(train_batch)

        losses = []
        for k, v in y_hat.items():
            if not k.startswith('predictions') or 'map' in k:
                continue
            l = bipartite_matching_segmentation_loss(v, train_batch['segmap'], power=self.power, pos_weight=self.pos_weight, neg_weight=self.neg_weight)
            self.log(f'train_bipartite_loss_{k}', l, on_step=False, on_epoch=True, reduce_fx=torch.sum)
            losses.append(l)

        if self.with_disp:
            l = F.smooth_l1_loss(y_hat['disp_pred'], train_batch['disparity'])
            self.log('train_disp_loss', l, on_step=False, on_epoch=True, reduce_fx=torch.sum)
            losses.append(l)

        self.confmat(y_hat['predictions_0'].clone().detach(), train_batch['segmap'])
        loss = sum(losses)
        self.log('train_loss', loss, on_step=False, on_epoch=True, reduce_fx=torch.sum)
        train_batch.update(y_hat)
        self.sample = train_batch
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.sample.tb(writer=self.logger.experiment[0], suffix='train', step=self.current_epoch)
        self.confmat.tb(writer=self.logger.experiment[0], descr=None, suffix='train', step=self.current_epoch)
        self.confmat.reset()

    def validation_step(self, val_batch, batch_idx):
        y_hat = self.forward(val_batch)

        losses = []
        for k, v in y_hat.items():
            if not k.startswith('predictions') or 'map' in k:
                continue
            l = bipartite_matching_segmentation_loss(v, val_batch['segmap'], power=self.power, pos_weight=self.pos_weight, neg_weight=self.neg_weight)
            self.log(f'val_bipartite_loss_{k}', l, on_step=False, on_epoch=True, reduce_fx=torch.sum)
            losses.append(l)

        if self.with_disp:
            l = F.smooth_l1_loss(y_hat['disp_pred'], val_batch['disparity'])
            self.log('val_disp_loss', l, on_step=False, on_epoch=True, reduce_fx=torch.sum)
            losses.append(l)

        self.confmat(y_hat['predictions_0'].clone().detach(), val_batch['segmap'])
        loss = sum(losses)
        self.log('val_loss', loss, on_step=False, on_epoch=True, reduce_fx=torch.sum)
        val_batch.update(y_hat)
        self.sample = val_batch

    def validation_epoch_end(self, outputs) -> None:
        self.sample.tb(writer=self.logger.experiment[0], suffix='val', step=self.current_epoch)
        self.confmat.tb(writer=self.logger.experiment[0], descr=None, suffix='val', step=self.current_epoch)
        self.confmat.reset()


if __name__ == '__main__':
    cfg = load_config('../configs/config.yaml')
    instr = INSTR(cfg=cfg).cuda()

    inp = torch.randn(2, 3, 480, 640).cuda()
    with torch.no_grad():
        out = instr.forward({"color_0": inp, "color_1": inp})

    for k, v in out.items():
        print(k, v.shape)
