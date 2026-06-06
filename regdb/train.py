from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData,LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from utils import *
from loss import OriTripletLoss, IntraModalCompactLoss,CenterTripletLoss,DCL,MSEL,CenterLoss,CenterAggregationLossWithGraph
from tensorboardX import SummaryWriter
import datetime
import torch.nn.functional as F
from torch.cuda import amp

from optimizer import make_optimizer
from scheduler import create_scheduler
import build_transforms

from resnet.model_resnet import net_resnet
from model.make_model import embed_net
from config.config import cfg
import torch.optim as optim


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',help='dataset name: regdb or sysu or llcm]')
parser.add_argument('--resume', '-r', default='sysu_LAReViT_p4_n6_lr_0.0003_seed_0_ADMW_best', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--save_epoch', default=20, type=int,metavar='s', help='save model every 10 epochs')
parser.add_argument('--optim', default='sgd', type=str, help='SGD,ADMW')
parser.add_argument('--model_path', default='result/LAReViT/save_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='result/LAReViT/log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='result/LAReViT/log/vis_log/', type=str, help='log save path')
parser.add_argument('--loss', default='CAL', type=str, help='')
parser.add_argument('--backbone', default='transformer',type=str, help='transformer')
parser.add_argument('--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.1 , type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default= 4, type=int,help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=int,help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--method', default='LAReViT', type=str,
                    metavar='m', help='method type: base or LAReViT')
parser.add_argument('--config_file', default='/root/autodl-tmp/LAReViT-master/config/SYSU.yml',
                    help='path to config file', type=str)
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None,nargs=argparse.REMAINDER)
parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
args = parser.parse_args()

if args.config_file != '':
    cfg.merge_from_file(args.config_file)

cfg.merge_from_list(args.opts)
cfg.freeze()

torch.cuda.set_device(args.gpu)
set_seed(args.seed)

#log
dataset = args.dataset
if dataset == 'sysu':
    data_path='/root/autodl-tmp/data/sysu/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [2, 1]  # thermal to visible
elif dataset == 'regdb':
    data_path = '/root/autodl-tmp/data/regdb/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
elif dataset == 'llcm':
    data_path = '/root/autodl-tmp/data/LLCM/'
    log_path = args.log_path + 'llcm_log/'
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
suffix = suffix + '_LAReViT_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, cfg.BASE_LR, args.seed)

if not args.optim == 'admw':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1
# test_mode = [1, 2]

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = build_transforms.test_transforms(
    args.img_h, args.img_w, normalize)
transform_color1 = build_transforms.train_transforms_color1(
    args.img_h, args.img_w, normalize)
transform_color2 = build_transforms.train_transforms_color2(
    args.img_h, args.img_w, normalize)
transform_thermal1 = build_transforms.train_transforms_thermal1(
    args.img_h, args.img_w, normalize)
transform_thermal2 = build_transforms.train_transforms_thermal2(
    args.img_h, args.img_w, normalize)
transform_train = transform_color1, transform_color2, transform_thermal1, transform_thermal2

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_dir=data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, args.img_h,
                         args.img_w, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(
        trainset.train_color_label, trainset.train_thermal_label)
    # testing set
    query_img, query_label = process_test_regdb(
        data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(
        data_path, trial=args.trial, modal='thermal')

elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, args.trial, args.img_h,
                        args.img_w, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(
        trainset.train_color_label, trainset.train_thermal_label)
    # testing set
    query_img, query_label, query_cam = process_query_llcm(
        data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(
        data_path, mode=test_mode[0], trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

gall_loader
print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))
print('==> Building model..')

net_vit = embed_net(args, n_class, no_local='on', gm_pool='on', arch=args.arch)
net_vit.to(device)


if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path,weights_only=False)
        start_epoch = checkpoint['epoch']
        net_vit.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion1 = nn.CrossEntropyLoss()
if args.method == 'LAReViT':
    loader_batch = args.batch_size * args.num_pos
    #criterion_tri= CenterTripletLoss(k_size=loader_batch, margin=args.margin)
    criterion2 = OriTripletLoss(batch_size=loader_batch, margin=args.margin)

else:
    loader_batch = args.batch_size * args.num_pos
    criterion2= OriTripletLoss(batch_size=loader_batch, margin=args.margin)



criterion1.to(device)
criterion2.to(device)
seq_lenth=1
##############################################

# optimizer
if args.optim == 'sgd':
    ignored_params =list(map(id, net_vit.bottleneck0.parameters()))+list(map(id, net_vit.bottleneck1.parameters()))+list(map(id, net_vit.bottleneck2.parameters())) + list(map(id, net_vit.bottleneck3.parameters())) + list(map(id, net_vit.bottleneck4.parameters())) \
                      + list(map(id, net_vit.classifier1.parameters()))+list(map(id, net_vit.classifier2.parameters())) + list(map(id, net_vit.classifier3.parameters())) + list(map(id, net_vit.classifier4.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net_vit.parameters())

    optimizer_P = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net_vit.bottleneck0.parameters(), 'lr': args.lr},
        {'params': net_vit.bottleneck1.parameters(), 'lr': args.lr},
        {'params': net_vit.bottleneck2.parameters(), 'lr': args.lr},
        {'params': net_vit.bottleneck3.parameters(), 'lr': args.lr},
        {'params': net_vit.bottleneck4.parameters(), 'lr': args.lr},
        {'params': net_vit.classifier1.parameters(), 'lr': args.lr},
        {'params': net_vit.classifier2.parameters(), 'lr': args.lr},
        {'params': net_vit.classifier3.parameters(), 'lr': args.lr},
        {'params': net_vit.classifier4.parameters(), 'lr': args.lr},
        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
def adjust_learning_rate(optimizer_P, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 35:
        lr = args.lr
    elif 35 <= epoch < 80:
        lr = args.lr * 0.1
    elif epoch >= 80:
        lr = args.lr * 0.01

    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    return lr

def compute_ide_loss(logits_list, pids):
    avg_ide_loss = 0
    avg_logits = 0
    part_num = 3
    for i in range(part_num):
        logits_i = logits_list[i]
        avg_logits += 1.0 / float(part_num) * logits_i
        ide_loss_i = criterion1(logits_i, pids)
        avg_ide_loss += 1.0 / float(part_num) * ide_loss_i
    return avg_ide_loss, avg_logits

def compute_part_loss(logits_list, labels_list):
    avg_ide_loss = 0
    avg_logits = 0
    part_num = 3
    for i in range(part_num):
        logits_i = logits_list[i]
        labels_list_i = labels_list[i]
        avg_logits += 1.0 / float(part_num) * logits_i
        ide_loss_i = criterion1(logits_i, labels_list_i)
        avg_ide_loss += 1.0 / float(part_num) * ide_loss_i
    return avg_ide_loss, avg_logits


def compute_part_loss_with_labels(logits_list, batch_size, device, criterion):
    num_parts = len(logits_list)

    labels_list = [
        torch.full((batch_size,), i, dtype=torch.long, device=device)
        for i in range(num_parts)
    ]

    losses = [criterion(logits_list[i], labels_list[i]) for i in range(num_parts)]
    avg_ide_loss = torch.stack(losses).mean()

    avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)

    return avg_ide_loss, avg_logits

#############################################
def train(epoch):
    current_lr = adjust_learning_rate(optimizer_P, epoch)
    train_loss = AverageMeter()
    id_loss0 = AverageMeter()
    tri_loss0 = AverageMeter()
    loss_ide0 = AverageMeter()

    id_loss1 = AverageMeter()
    tri_loss1 = AverageMeter()
    id_loss2 = AverageMeter()
    tri_loss2 = AverageMeter()
    id_loss3 = AverageMeter()
    tri_loss3 = AverageMeter()
    id_loss_a = AverageMeter()
    part_loss = AverageMeter()
    kl_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    id_loss4 = AverageMeter()
    tri_loss4 = AverageMeter()
    align_meter = AverageMeter()

    correct = 0
    total = 0

    net_vit.train()
    end = time.time()

    for batch_idx, (input10, input11, input20, input21, label1, label2) in enumerate(trainloader):
        with amp.autocast(enabled=True):
            labels = torch.cat((label1, label2), 0)

            input10 = Variable(input10.cuda())
            input11 = Variable(input11.cuda())
            input20 = Variable(input20.cuda())
            input21 = Variable(input21.cuda())


            labels = Variable(labels.cuda())
            label1 = Variable(label1.cuda())
            label2 = Variable(label2.cuda())
            data_time.update(time.time() - end)

            feat, out0 , de_feature, de_feature_p, spatial_feat, spatial_feat_p, logits_list1,  fused_feat,fused_feat_p,align_loss= net_vit(input10, input20, seq_len=seq_lenth)

            # add
            loss_id0 = criterion1(out0, labels)
            loss_id2 = criterion1(de_feature_p, labels)
            loss_id3 = criterion1(spatial_feat_p, labels)
            loss_id4 = criterion1(fused_feat_p, labels)

            # loss_part, avg_logits1 = compute_part_loss_with_labels(logits_list1, labels_list)
            loss_part, avg_logits1 = compute_part_loss_with_labels(
                logits_list1,
                batch_size=loader_batch * 2,
                device=logits_list1[0].device,
                criterion=criterion1
            )
            loss_tri0 = criterion2(feat, labels)
            loss_tri2= criterion2(de_feature, labels)
            loss_tri3  = criterion2(spatial_feat, labels)
            loss_tri4 = criterion2(fused_feat, labels)

            # correct += (batch_acc0 / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(labels).sum().item() / 2)

            # loss function
            if epoch <= 40:
                loss = (loss_id0 + loss_tri0) + loss_part + align_loss
            else:
                loss = (loss_id0 + loss_tri0 + loss_id2 + loss_tri2) + \
                       (loss_id4 + loss_tri4) + (loss_id3 + loss_tri3) + align_loss + loss_part

        optimizer_P.zero_grad()
        loss.backward()
        optimizer_P.step()

        train_loss.update(loss.item(), 2 * input10.size(0))
        id_loss0.update(loss_id0.item(), 2 * input10.size(0))
        tri_loss0.update(loss_tri0.item(), 2 * input10.size(0))

        id_loss2.update(loss_id2.item(), 2 * input10.size(0))
        tri_loss2.update(loss_tri2.item(), 2 * input10.size(0))

        id_loss3.update(loss_id3.item(), 2 * input10.size(0))
        tri_loss3.update(loss_tri3.item(), 2 * input10.size(0))

        id_loss4.update(loss_id4.item(), 2 * input10.size(0))
        tri_loss4.update(loss_tri4.item(), 2 * input10.size(0))

        part_loss.update(loss_part.item(), 2 * input10.size(0))
        align_meter.update(align_loss.item(), 2 * input10.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:
            print(
                'Epoch: [{}][{}/{}] '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'lr:{:.6f} '
                'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) | '
                'ID0: {id0.val:.3f} ({id0.avg:.3f})  Tri0: {tri0.val:.3f} ({tri0.avg:.3f}) | '
                'ID2: {id2.val:.3f} ({id2.avg:.3f})  Tri2: {tri2.val:.3f} ({tri2.avg:.3f}) | '
                'ID3: {id3.val:.3f} ({id3.avg:.3f})  Tri3: {tri3.val:.3f} ({tri3.avg:.3f}) | '
                'ID4: {id4.val:.3f} ({id4.avg:.3f})  Tri4: {tri4.val:.3f} ({tri4.avg:.3f}) | '
                'Part: {part.val:.3f} ({part.avg:.3f})  Align: {align.val:.3f} ({align.avg:.3f}) | '
                'Accu: {:.2f}'
                .format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total,
                    batch_time=batch_time,
                    train_loss=train_loss,
                    id0=id_loss0, tri0=tri_loss0,
                    id2=id_loss2, tri2=tri_loss2,
                    id3=id_loss3, tri3=tri_loss3,
                    id4=id_loss4, tri4=tri_loss4,
                    part=part_loss, align=align_meter
                )
            )

    writer.add_scalar('id_loss2', id_loss2.avg, epoch)
    writer.add_scalar('tri_loss2', tri_loss2.avg, epoch)
    writer.add_scalar('id_loss3', id_loss3.avg, epoch)
    writer.add_scalar('tri_loss3', tri_loss3.avg, epoch)
    writer.add_scalar('id_loss4', id_loss4.avg, epoch)
    writer.add_scalar('tri_loss4', tri_loss4.avg, epoch)
    writer.add_scalar('part_loss', part_loss.avg, epoch)
    # writer.add_scalar('align_loss', align_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    return 1. / (1. + train_loss.avg)

HTTT=2048
def test(epoch):
    # switch to evaluation mode
    net_vit.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, HTTT))
    gall_feat_att = np.zeros((ngall, HTTT))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())

            featA = net_vit(input, input, test_mode[1], seq_len=seq_lenth)

            gall_feat[ptr:ptr + batch_num, :] = featA.detach().cpu().numpy()
            # gall_feat_att[ptr:ptr + batch_num, :] = featA1.detach().cpu().numpy()

            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net_vit.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, HTTT))
    query_feat_att = np.zeros((nquery, HTTT))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            featA = net_vit(input, input, test_mode[0], seq_len=seq_lenth)

            query_feat[ptr:ptr + batch_num, :] = featA.detach().cpu().numpy()
            # query_feat_att[ptr:ptr + batch_num,
            #                :] = featA1.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()

    distmat_att = np.matmul(query_feat, np.transpose(gall_feat))
    # distmat = np.matmul(query_feat_att, np.transpose(gall_feat_att))
    # evaluation
    if dataset == 'regdb':
        # cmc, mAP, mINP  = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        # cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'llcm':
        # cmc, mAP, mINP = eval_llcm(-distmat, query_label,
        #                            gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(
            -distmat_att, query_label, gall_label, query_cam, gall_cam)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    # writer.add_scalar('rank1', cmc[0], epoch)
    # writer.add_scalar('mAP', mAP, epoch)
    # writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return  cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, 201 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)


    # training
    train(epoch)

    if epoch >=0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net_vit.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': net_vit.state_dict(),
                # 'cmc': cmc,
                # 'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        # print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        #     cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))