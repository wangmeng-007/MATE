"""VSE model"""
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder, FuseTransEncoder
from lib.loss import ContrastiveLoss, TransformerContrastiveLoss

import logging

logger = logging.getLogger(__name__)

class VSEModel(object):
    """
        The standard VSE model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(opt.data_name, opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)
        self.transformer_encoder = FuseTransEncoder(num_layers=opt.num_layers,hidden_size=opt.embed_size * 2,nhead=opt.nhead, dropout=0.1)

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        #self.transformer_criterion = TransformerContrastiveLoss(batch_size=opt.batch_size, device='cuda:0', temperature=0.5)
        self.transformer_criterion = TransformerContrastiveLoss(batch_size=opt.batch_size, margin=1.0,
                                                                temperature=0.5,num_hard_negatives=1,lambda_contrastive=0.5,)

        # Introduce momentum models for ESA and Transformer
        self.img_enc_m = copy.deepcopy(self.img_enc)
        self.txt_enc_m = copy.deepcopy(self.txt_enc)
        self.transformer_encoder_m = copy.deepcopy(self.transformer_encoder)

        self.momentum = getattr(opt, 'momentum', 0.995)
        self.transformer_lr = getattr(opt, 'transformer_lr', 0.0005)

        # Disable gradients for momentum models
        for param in self.img_enc_m.parameters():
            param.requires_grad = False
        for param in self.txt_enc_m.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder_m.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.transformer_encoder.cuda()
            self.transformer_criterion.cuda()
            self.img_enc_m.cuda()
            self.txt_enc_m.cuda()
            self.transformer_encoder_m.cuda()
            cudnn.benchmark = True

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.transformer_encoder.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.transformer_encoder.parameters(), 'lr':opt.learning_rate}
                ],
                    lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))
        else:
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.backbone.top.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.img_enc.backbone.base.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                    {'params': self.transformer_encoder.parameters(), 'lr': opt.learning_rate}
                ], lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD([
                    {'params': self.txt_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.img_enc.backbone.parameters(), 'lr': opt.learning_rate * opt.backbone_lr_factor,
                     'weight_decay': decay_factor},
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                    {'params': self.transformer_encoder.parameters(),'lr':opt.learning_rate}
                ], lr=opt.learning_rate, momentum=0.9, nesterov=True)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.transformer_encoder.state_dict(),
        self.img_enc_m.state_dict(),  # momentum model state
        self.txt_enc_m.state_dict(),
        self.transformer_encoder_m.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        """加载模型参数，确保兼容旧版本没有动量模型的模型文件"""

        # 加载基础模型的参数
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)
        self.transformer_encoder.load_state_dict(state_dict[2], strict=False)

        # 兼容没有动量模型的旧模型
        if len(state_dict) > 3:
            # 如果保存了动量模型的参数，加载动量模型的参数
            self.img_enc_m.load_state_dict(state_dict[3], strict=False)
            self.txt_enc_m.load_state_dict(state_dict[4], strict=False)
            self.transformer_encoder_m.load_state_dict(state_dict[5], strict=False)
        else:
            print("Warning: No momentum model found in the checkpoint. Only loading base models.")

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.transformer_encoder.train()
        self.img_enc_m.train()
        self.txt_enc_m.train()
        self.transformer_encoder_m.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.transformer_encoder.eval()
        self.img_enc_m.eval()
        self.txt_enc_m.eval()
        self.transformer_encoder_m.eval()

    def freeze_backbone(self):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.freeze_backbone()
            else:
                self.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.unfreeze_backbone(fixed_blocks)
            else:
                self.img_enc.unfreeze_backbone(fixed_blocks)

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.transformer_encoder = nn.DataParallel(self.transformer_encoder)
        self.img_enc_m = nn.DataParallel(self.img_enc_m)
        self.txt_enc_m = nn.DataParallel(self.txt_enc_m)
        self.transformer_encoder_m = nn.DataParallel(self.transformer_encoder_m)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    @torch.no_grad()
    def copy_params(self):
        """将基础模型的参数复制到动量模型"""
        for param, param_m in zip(self.img_enc.parameters(), self.img_enc_m.parameters()):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False

        for param, param_m in zip(self.txt_enc.parameters(), self.txt_enc_m.parameters()):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False

        for param, param_m in zip(self.transformer_encoder.parameters(), self.transformer_encoder_m.parameters()):
            param_m.requires_grad = False
            param_m.data.copy_(param.data)

    @torch.no_grad()
    def update_momentum(self):
        """更新动量模型的参数"""
        for param, param_m in zip(self.img_enc.parameters(), self.img_enc_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

        for param, param_m in zip(self.txt_enc.parameters(), self.txt_enc_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

        for param, param_m in zip(self.transformer_encoder.parameters(), self.transformer_encoder_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def forward_emb(self, images, captions, lengths, image_lengths=None):
        """Compute the image and caption embeddings for both base and momentum models
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Compute base model embeddings
        img_emb = self.img_enc(images)
        lengths = torch.Tensor(lengths).cuda()  # (256,)
        cap_emb = self.txt_enc(captions, lengths)

        # Compute momentum model embeddings
        with torch.no_grad():
            img_emb_m = self.img_enc_m(images)
            cap_emb_m = self.txt_enc_m(captions, lengths)

        return img_emb, cap_emb, img_emb_m, cap_emb_m

    def forward_loss(self, img_emb, cap_emb, img_emb_m, cap_emb_m):
        """Compute the loss given pairs of image and caption embeddings for both base and momentum models."""

        # Compute base model contrastive loss
        cost_im, cost_s = self.criterion(img_emb, cap_emb)
        self.logger.update('Loss_im', cost_im.item(), cap_emb.size(0))
        self.logger.update('Loss_s', cost_s.item(), cap_emb.size(0))

        # Normalize embeddings for computing KL divergence
        img_emb_norm = F.normalize(img_emb, dim=-1)
        cap_emb_norm = F.normalize(cap_emb, dim=-1)
        img_emb_m_norm = F.normalize(img_emb_m, dim=-1)
        cap_emb_m_norm = F.normalize(cap_emb_m, dim=-1)

        # Compute KL divergence loss between base model and momentum model embeddings
        distill_loss_im_kl = F.kl_div(
            F.log_softmax(img_emb_norm, dim=-1),
            F.softmax(img_emb_m_norm, dim=-1),
            reduction='batchmean'
        )
        distill_loss_cap_kl = F.kl_div(
            F.log_softmax(cap_emb_norm, dim=-1),
            F.softmax(cap_emb_m_norm, dim=-1),
            reduction='batchmean'
        )

        # Combine the KL divergence losses for both image and caption embeddings
        distill_loss = distill_loss_im_kl + distill_loss_cap_kl
        self.logger.update('Le0', distill_loss.item(), cap_emb.size(0))

        # Combine the base contrastive loss with the KL divergence distillation loss
        loss = cost_im + cost_s+ 0.5 * distill_loss
        self.logger.update('Le', loss.item(), cap_emb.size(0))

        return loss

    def train_emb(self, images, captions, lengths, image_lengths=None, warmup_alpha=None):
        """One training step given images and captions, with momentum model integration.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        captions_all = captions.reshape(captions.size(0) * captions.size(1), captions.size(2))
        caption_lens = lengths.reshape(-1)

        # Compute the embeddings for base and momentum models
        img_emb, cap_emb, img_emb_m, cap_emb_m = self.forward_emb(images, captions_all, caption_lens,
                                                                  image_lengths=image_lengths)

        # Measure accuracy and record loss
        self.optimizer.zero_grad()
        loss_esa = self.forward_loss(img_emb, cap_emb, img_emb_m, cap_emb_m)

        if warmup_alpha is not None:
            loss_esa = loss_esa * warmup_alpha

        # Compute gradient and update
        loss_esa.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        # Update momentum models
        self.update_momentum()

    def train_fusion(self, images, captions, lengths, img_lengths=None, warmup_alpha=None):
        """Train the transformer encoder with momentum model integration"""
        self.transformer_encoder.train()
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # Compute embeddings from ESA modules (base model)
        img_emb = self.img_enc(images)  # (256,1024)

        # Check and adjust lengths for captions
        if lengths.dim() == 2 and lengths.size(1) == 2:
            lengths = lengths.view(-1).cuda()
        else:
            lengths = torch.Tensor(lengths).cuda()
        if captions.dim() == 3:
            captions = captions.view(-1, captions.size(-1))

        cap_emb = self.txt_enc(captions, lengths)  # (256,1024)

        # Concatenate image and text embeddings
        combined_features = torch.cat((img_emb, cap_emb), dim=1).to(img_emb.device)  # (256,2048)
        combined_features = combined_features.unsqueeze(0)  # (1,256,2048)

        # Compute transformer output for base model
        img_emb, cap_emb = self.transformer_encoder(combined_features)

        # Compute embeddings for the momentum model
        with torch.no_grad():
            img_emb_m = self.img_enc_m(images)
            cap_emb_m = self.txt_enc_m(captions, lengths)
            combined_features_m = torch.cat((img_emb_m, cap_emb_m), dim=1).to(img_emb.device).unsqueeze(0)
            img_emb_m, cap_emb_m = self.transformer_encoder_m(combined_features_m)

        # Compute negative samples for contrastive loss
        neg_emb = torch.roll(cap_emb, shifts=1, dims=0)  # (256,1024)

        # Compute the contrastive loss for the base model
        loss_transformer = self.transformer_criterion(img_emb, cap_emb, neg_emb)

        # Compute momentum distillation loss using KL divergence
        distill_loss_img_kl = F.kl_div(F.log_softmax(img_emb, dim=-1), F.softmax(img_emb_m, dim=-1),
                                       reduction='batchmean')
        distill_loss_cap_kl = F.kl_div(F.log_softmax(cap_emb, dim=-1), F.softmax(cap_emb_m, dim=-1),
                                       reduction='batchmean')

        # Combine distillation losses (KL divergence)
        distill_loss = distill_loss_img_kl + distill_loss_cap_kl

        # Combine the two losses (contrastive loss and distillation loss)
        loss = loss_transformer + 0.5 * distill_loss
        self.logger.update('Le1', loss_transformer.item(), cap_emb.size(0))
        self.logger.update('Le2', distill_loss.item(), cap_emb.size(0))

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        # Update momentum models
        self.update_momentum()
