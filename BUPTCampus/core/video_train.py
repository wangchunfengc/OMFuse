import torch
import torch.nn.functional as F
import copy
from torch.cuda import amp
from tools.loss import diversity_loss

def process_vedio(x1,seq_len=10):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1


def compute_part_loss(base,logits_list, labels_list):
    avg_ide_loss = 0
    avg_logits = 0
    part_num = 3
    for i in range(part_num):
        logits_i = logits_list[i]
        labels_list_i = labels_list[i]
        avg_logits += 1.0 / float(part_num) * logits_i
        ide_loss_i = base.criterion1(logits_i, labels_list_i)
        avg_ide_loss += 1.0 / float(part_num) * ide_loss_i
    return avg_ide_loss, avg_logits


def foward_video(iter,base,meter,scaler,epoch):
    # scaler = amp.GradScaler()
    for _ in range(base.steps):
        input1, input2, label1, label2 = iter.next_one()
        base.model_optimizer.zero_grad()
        rgb_imgs, rgb_pids = input1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs,  rgb_pids = rgb_imgs.to(base.device),rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()
        rgb_imgs = process_vedio(rgb_imgs,base.seq_lenth)
        ir_imgs = process_vedio(ir_imgs,base.seq_lenth)

        labels = torch.cat([rgb_pids, ir_pids], dim=0)

        with (amp.autocast(enabled=True)):
            # features, cls_score = base.model(x1=rgb_imgs, x2=ir_imgs)
            # print(rgb_imgs.shape,"---------------")                     torch.Size([160, 3, 288, 144])
            feat, out0 , de_feature, de_feature_p, spatial_feat, spatial_feat_p, logits_list1,  fused_feat,fused_feat_p,align_loss= base.model(
                x1=rgb_imgs, x2=ir_imgs, modal=0, seq_len=base.seq_lenth
            )

            criterion_id = base.criterion1  # CrossEntropy
            criterion_tri = base.criterion2  # Triplet

            loss_id0 = criterion_id(out0, labels)
            loss_id2 = criterion_id(de_feature_p, labels)
            loss_id3 = criterion_id(spatial_feat_p, labels)
            loss_id4 = criterion_id(fused_feat_p, labels)

            labels_list = [labels for _ in range(len(logits_list1))]
            loss_part, _ = compute_part_loss(base,logits_list1, labels_list)

            loss_tri0, _ = criterion_tri(feat, labels)
            loss_tri2, _ = criterion_tri(de_feature, labels)
            loss_tri3, _ = criterion_tri(spatial_feat, labels)
            loss_tri4, _ = criterion_tri(fused_feat, labels)

            if epoch <= 40:
                total_loss = (loss_id0 + loss_tri0 ) + loss_part + align_loss
            else:
                total_loss =  (loss_id0 + loss_tri0 + loss_id2 + loss_tri2 )+\
            (loss_id4 + loss_tri4)  +(loss_id3 + loss_tri3) + align_loss + loss_part


            # total_loss = ide_loss + ide_loss_proj + triplet_loss_last + triplet_loss + triplet_loss_proj + ide_loss_text + ide_loss_cue + triplet_loss_cue
        scaler.scale(total_loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()

        meter.update({
            'loss_ce_global': loss_id0.item(),
            'loss_ce_decoder': loss_id2.item(),
            'loss_ce_fused': loss_id4.item(),
            'loss_tri_global': loss_tri0.item(),
            'loss_tri_decoder': loss_tri2.item(),
            'loss_tri_fused': loss_tri4.item(),
            'loss_align': (align_loss.item() if hasattr(align_loss, 'item') else float(align_loss)),
            'loss_total': total_loss.item(),
        })
    return meter