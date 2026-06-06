import copy
import torch
import torch.nn as nn
from torch.nn import init
from model.resnet import resnet50, resnet18
from model.clip_model import Transformer
import torch.nn.functional as F



class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)





class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.layer4 = copy.deepcopy(self.base.layer4)

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        # t_x = self.layer4(x)
        x = self.base.layer4(x)
        return x


def conv1x1(conv, x):
    x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
    x = conv(x)
    x = x.squeeze()
    return x


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z



class SpatialFeature(nn.Module):
    def __init__(self, args, num_stripes, embed_dim, seq_len, class_num):
        super(SpatialFeature, self).__init__()
        self.num_part = num_stripes
        self.embed_dim = embed_dim
        self.class_num = class_num
        self.seq_len = seq_len

        # 分类器
        for i in range(self.num_part):
            setattr(self, f'classifier{i}', nn.Linear(embed_dim, class_num))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.TransLocal = Transformer(width=self.embed_dim, layers=args.cmt_depth,
                                      heads=self.embed_dim // 64)
        self.TransGlobal = Transformer(width=self.embed_dim, layers=args.cmt_depth,
                                       heads=self.embed_dim // 64)

        self.TransLocal_paired = Transformer(width=self.embed_dim, layers=args.cmt_depth,
                                             heads=self.embed_dim // 64)

        self.TransLocal_reverse = Transformer(width=self.embed_dim, layers=args.cmt_depth,
                                              heads=self.embed_dim // 64)


    def forward(self, x):
        B, C, H, W = x.shape
        stripe_h = int(H / self.num_part)

        single_feats = []
        paired_feats = []
        single_weights = []
        paired_weights = []

        Avglocal_feat = []
        paired_Avglocal_feat = []

        for i in range(self.num_part):
            part = x[:, :, i * stripe_h:(i + 1) * stripe_h, :]              #(b*t,c,h,w)
            feat = self.avgpool(part).squeeze()                             #(b*t,c)
            feat = feat.view(B // self.seq_len, self.seq_len, -1)           #(b,t,c)
            single_feats.append(feat)

            avg_feat = torch.mean(feat, dim=1)                              #(b,c)
            Avglocal_feat.append(avg_feat)
            feat_perm = feat.permute(1, 0, 2)                               #(t,b,c)
            attention = torch.mul(feat_perm, Avglocal_feat[i]).softmax(dim=-1) #(t,b,c)
            weighted = torch.mul(attention, feat_perm).sum(dim=0)           #(b,c)
            single_weights.append(weighted)

        for i in range(self.num_part):
            j = (i + 1) % self.num_part
            paired = torch.cat([
                x[:, :, i * stripe_h:(i + 1) * stripe_h, :],
                x[:, :, j * stripe_h:(j + 1) * stripe_h, :]                 #(b*t,c,h,w)
            ], dim=2)

            feat = self.avgpool(paired).squeeze()                           #(b*t,c)
            feat = feat.view(B // self.seq_len, self.seq_len, -1)           #(b,t,c)
            paired_feats.append(feat)

            avg_feat = torch.mean(feat, dim=1)                              #(b,c)
            paired_Avglocal_feat.append(avg_feat)
            feat_perm = feat.permute(1, 0, 2)                               #(t,b,c)
            attention = torch.mul(feat_perm, paired_Avglocal_feat[i]).softmax(dim=-1) #(t,b,c)
            weighted = torch.mul(attention, feat_perm).sum(dim=0)                       #(b,c)
            paired_weights.append(weighted)

        final_feats = []
        for i in range(self.num_part):
            feat_forward = self.TransLocal(single_feats[i])                 #(b,t,c)
            feat_forward = torch.mean(feat_forward, dim=1)                  #(b,c)

            feat_reverse = self.TransLocal_reverse(torch.flip(single_feats[i], dims=[1]))
            feat_reverse = torch.mean(feat_reverse, dim=1)

            feat_paired = self.TransLocal_paired(paired_feats[i])
            feat_paired = torch.mean(feat_paired, dim=1)

            fused_feat = feat_forward + feat_reverse + feat_paired + single_weights[i] + paired_weights[i]
            final_feats.append(fused_feat)

        global_feat = torch.stack(final_feats, dim=1)        #（b,n,t）
        global_feat = self.TransGlobal(global_feat)
        global_feat = torch.mean(global_feat, dim=1)

        logits_list = []
        for i in range(self.num_part):
            classifier = getattr(self, f'classifier{i}')
            logits_list.append(classifier(final_feats[i]))

        return global_feat, logits_list


class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(FiLM, self).__init__()

        self.fc_gamma = nn.Linear(input_dim, condition_dim)
        self.fc_beta = nn.Linear(input_dim, condition_dim)

    def forward(self, x, condition):
        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)
        y = gamma * x + beta
        return y

class embed_net(nn.Module):
    def __init__(self, args, class_num, drop=0.2, no_local='on', gm_pool='on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        pool_dim = 2048
        self.embed_dim = pool_dim
        self.dropout = drop
        self.non_local = no_local
        self.gm_pool = gm_pool


        self.l2norm = Normalize(2)

        self.bottleneck0 = nn.BatchNorm1d(pool_dim)
        self.bottleneck0.bias.requires_grad_(False)  # no shift
        self.bottleneck0.apply(weights_init_kaiming)

        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)

        self.bottleneck2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)
        self.bottleneck3 = nn.BatchNorm1d(pool_dim)
        self.bottleneck3.bias.requires_grad_(False)
        self.bottleneck3.apply(weights_init_kaiming)
        self.bottleneck4 = nn.BatchNorm1d(pool_dim)
        self.bottleneck4.bias.requires_grad_(False)
        self.bottleneck4.apply(weights_init_kaiming)
        #add
        # self.bottleneck5 = nn.BatchNorm1d(pool_dim)
        # self.bottleneck5.bias.requires_grad_(False)
        # self.bottleneck5.apply(weights_init_kaiming)

        # self.classifier0 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier3 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier4 = nn.Linear(pool_dim, class_num, bias=False)
        # self.classifier5 = nn.Linear(pool_dim, class_num, bias=False)


        # self.classifier0.apply(weights_init_classifier)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)
        self.classifier3.apply(weights_init_classifier)
        self.classifier4.apply(weights_init_classifier)
        # self.classifier5.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        # self.encoder = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)
        self.decoder = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)

        self.SpatialFeature = SpatialFeature(args, num_stripes=3, embed_dim=pool_dim,  seq_len=1, class_num=class_num)

        self.film = FiLM(2048,2048)
        self.Temporal_ir = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)
        self.Temporal_vis = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)

        self.norm1 = nn.LayerNorm(pool_dim)
        self.norm2 = nn.LayerNorm(pool_dim)
        self.norm3 = nn.LayerNorm(pool_dim)
        self.norm4 = nn.LayerNorm(pool_dim)

        self.fc_gate = nn.Linear(2 * pool_dim, pool_dim)
        nn.init.zeros_(self.fc_gate.weight)
        nn.init.zeros_(self.fc_gate.bias)

        self.fuse_head = nn.LayerNorm(pool_dim)

        self.cls_part = nn.Linear(self.embed_dim, 3, bias=False)
        self.cls_part.apply(weights_init_classifier)

    def forward(self, x1, x2, modal=0, seq_len=6):
        b, c, h, w = x1.size()
        t = seq_len
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x = self.base_resnet(x)

        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x_ = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x_ ** p, dim=-1) + 1e-12) ** (1 / p)
            x_pool = x_pool.view(x_pool.size(0) // t, t, -1)


            spatial_feat, logits_list1 = self.SpatialFeature(x)

            if self.training:
                visible = x[:b // 2]
                infrared = x[b // 2:]
                B, C, H, W = visible.shape

                # [B*T, C, 1, 1] -> [B*T, C] -> [B, T, C]
                visible_feat = self.avgpool(visible).squeeze(-1).squeeze(-1)
                visible_feat = visible_feat.view(B // seq_len, seq_len, -1)  # [B,T,C]

                infrared_feat = self.avgpool(infrared).squeeze(-1).squeeze(-1)
                infrared_feat = infrared_feat.view(B // seq_len, seq_len, -1)  # [B,T,C]

            if modal == 0:
                f1 = self.film(infrared_feat, visible_feat)  # [B,T,C]
                f1 = f1.mean(dim=1)  # [B,C]

                tau = 2.0
                f_vis = self.Temporal_vis(self.norm1(visible_feat))
                vis_alpha = torch.softmax(f_vis.mean(-1)/tau, dim=1).unsqueeze(-1)  # [B,T,1]
                f2 = (vis_alpha * self.norm2(visible_feat)).sum(1)

                f_ir = self.Temporal_ir(self.norm3(infrared_feat))
                ir_alpha = torch.softmax(f_ir.mean(-1)/tau, dim=1).unsqueeze(-1)
                f3 = (ir_alpha * self.norm4(infrared_feat)).sum(1)

                gate_ir = torch.sigmoid(self.fc_gate(torch.cat([f1, f2], dim=1)))  # [B, C]
                fused_ir = gate_ir * f1 + (1 - gate_ir) * f2

                gate_vis = torch.sigmoid(self.fc_gate(torch.cat([f2, f3], dim=1)))  # [B, C]
                fused_vis = gate_vis * f2 + (1 - gate_vis) * f3

                fused_all = torch.cat([fused_vis, fused_ir], dim=0)  # [2B, C]
                fused_all = self.fuse_head(fused_all)

                fused_feat_for_tri = F.normalize(fused_all, dim=1)
                fused_logits = self.classifier1(fused_all)

                fused_feat = fused_all

                B2 = fused_feat_for_tri.size(0) // 2
                vis_align = fused_feat_for_tri[:B2]  # [B, C]
                ir_align = fused_feat_for_tri[B2:]  # [B, C]
                align_loss = 1.0 - F.cosine_similarity(vis_align, ir_align, dim=1).mean()


            elif modal == 1 or modal == 2:
                x = self.avgpool(x).squeeze(-1).squeeze(-1)
                x = x.view(b // seq_len, seq_len, -1)  # [B, T, C]
                tau = 2.0
                if modal == 1:
                    z = self.Temporal_vis(self.norm1(x))  # [B,T,C]
                    alpha = torch.softmax(z.mean(-1) / tau, dim=1).unsqueeze(-1)  # [B,T,1]
                    y = (alpha * self.norm2(x)).sum(1)  # [B,C]
                elif modal==2:
                    z = self.Temporal_ir(self.norm3(x))
                    alpha = torch.softmax(z.mean(-1) / tau, dim=1).unsqueeze(-1)
                    y = (alpha * self.norm4(x)).sum(1)

                fused_all = self.fuse_head(y)  # LN
                fused_feat_for_tri = F.normalize(fused_all, dim=1)
                fused_logits = self.classifier1(fused_all)
                fused_feat = fused_all
            feat = torch.mean(x_pool, dim=1)
            de_feature = self.decoder(x_pool)
            de_feature = torch.mean(de_feature, dim=1)
            feat2_norm1d = self.bottleneck2(de_feature)
            feat_norm1d = self.bottleneck3(feat)
            spatial_feat_norm1d = self.bottleneck4(spatial_feat)


        if self.training:
            return  feat, self.classifier3(feat_norm1d),de_feature, self.classifier2(feat2_norm1d), spatial_feat, self.classifier4(spatial_feat_norm1d), logits_list1,fused_feat_for_tri, fused_logits ,align_loss
        else:
            return self.l2norm(feat2_norm1d +fused_feat+spatial_feat_norm1d)

