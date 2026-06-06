import os
import torch
import torch.nn as nn
# from network import  Classifier,Model
from tools import os_walk, TripletLoss_WRT,SupConLoss,OriTripletLoss
from network.scheduler import CosineLRScheduler

from network.model_main import embed_net
import torch.optim as optim

class Base:
    def __init__(self, config):
        self.config = config
        self.pid_num = config.pid_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')

        self.learning_rate = config.learning_rate
        self.lr_times = config.lr_times
        self.weight_decay = config.weight_decay
        self.steps = config.steps

        self.img_h = config.img_h
        self.img_w = config.img_w
        self.seq_lenth = config.sequence_length
        self.weight_a = config.weight_a
        self.weight_b = config.weight_b
        self.weight_c = config.weight_c
        self.STH_start_layer = config.STH_start_layer

        self._init_device()
        self._init_model()
        self._init_creiteron()
        self._init_optimizer()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):
        # self.model = Model(self.config,self.pid_num, self.img_h, self.img_w)
        self.model = embed_net(self.config, self.pid_num)
        self.model = self.model.to(self.device)


    def _init_creiteron(self):
        self.criterion1 = nn.CrossEntropyLoss()                 #id
        loader_batch = self.config.batch_size * self.config.num_pos
        self.criterion2 = OriTripletLoss(batch_size=loader_batch, margin=self.config.margin)
        # self.con_creiteron = SupConLoss(self.device)


    # def _init_optimizer(self):
    #     params = []
    #     keys = []
    #     for key, value in self.model.named_parameters():
    #         if "text_encoder" in key:
    #             value.requires_grad_(False)
    #             continue
    #         if not value.requires_grad:
    #             continue
    #         lr = self.learning_rate
    #         weight_decay = self.weight_decay
    #         if "bias" in key:
    #             lr = self.learning_rate * self.lr_times
    #             weight_decay = self.weight_decay
    #
    #         if "classifier" in key or "arcface" in key:
    #             lr = self.learning_rate * self.lr_times
    #
    #
    #         if "prompt_learner" in key:
    #             lr = self.learning_rate * self.lr_times
    #
    #
    #         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    #         keys += [key]
    #
    #     self.model_optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
    #     lr_min = self.learning_rate*0.01
    #     warmup_lr_init = self.learning_rate*0.01
    #     warmup_t = 3
    #     self.model_lr_scheduler = CosineLRScheduler(
    #         self.model_optimizer,
    #         t_initial=24,
    #         lr_min=lr_min,
    #         t_mul= 1.,
    #         decay_rate=1.0,
    #         warmup_lr_init=warmup_lr_init,
    #         warmup_t=warmup_t,
    #         cycle_limit=1,
    #         t_in_epochs=True,
    #         noise_range_t=None,
    #         noise_pct= 0.67,
    #         noise_std= 1.,
    #         noise_seed=42,
    #     )
    def _init_optimizer(self):


        net = self.model
        args = self.config

        ignored_params = list(map(id, net.bottleneck0.parameters())) \
                         + list(map(id, net.bottleneck1.parameters())) \
                         + list(map(id, net.bottleneck2.parameters())) \
                         + list(map(id, net.bottleneck3.parameters())) \
                         + list(map(id, net.bottleneck4.parameters())) \
                         + list(map(id, net.classifier1.parameters())) \
                         + list(map(id, net.classifier2.parameters())) \
                         + list(map(id, net.classifier3.parameters())) \
                         + list(map(id, net.classifier4.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer_P = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.learning_rate},
            {'params': net.bottleneck0.parameters(), 'lr': args.learning_rate},
            {'params': net.bottleneck1.parameters(), 'lr': args.learning_rate},
            {'params': net.bottleneck2.parameters(), 'lr': args.learning_rate},
            {'params': net.bottleneck3.parameters(), 'lr': args.learning_rate},
            {'params': net.bottleneck4.parameters(), 'lr': args.learning_rate},
            {'params': net.classifier1.parameters(), 'lr': args.learning_rate},
            {'params': net.classifier2.parameters(), 'lr': args.learning_rate},
            {'params': net.classifier3.parameters(), 'lr': args.learning_rate},
            {'params': net.classifier4.parameters(), 'lr': args.learning_rate},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        self.model_optimizer = optimizer_P

    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, 'best_model.pth'.format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)

    def resume_model(self, resume_path):
        model_path = os.path.join(resume_path, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print('Successfully resume shared_model from {}'.format(model_path))


    def set_train(self):
        self.model = self.model.train()
        self.training = True

    def set_eval(self):
        self.model = self.model.eval()
        self.training = False





