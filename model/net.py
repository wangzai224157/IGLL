import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from model.basenet import BaseNet
from model.loss import VGGLoss , svd_loss , TVLoss
from model.layer import init_weights, ConfidenceDrivenMaskLayer
import numpy as np
from util.utils import generate_mask
from functools import reduce
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn

from model.Attention import GlobalLocalAttention, GlobalAttention
import datetime  

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_dim = 5
        self.cnum = 32
        self.corse_dim=7

        self.coarse_generator = CoarseGenerator(self.corse_dim, self.cnum)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum )

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2 = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2


class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()


        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 2, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum * 2, cnum * 4, 3, 1, 1)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16)
        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, activation='relu')
        self.contextul_attention = GlobalAttention(in_dim=128)
        self.pmconv9 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum * 8, cnum * 4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum // 2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum // 2, 3, 3, 1, 1, activation='none')

    def forward(self, xin, mask):
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3)).cuda()
        #xnow = torch.cat([xin, mask], dim=1)
        # conv branch
        xnow = torch.cat([xin, ones], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x = self.contextul_attention(x, x, mask)
       
  
        """"
        
        
        res = self.contextul_attention(x, x, mask)
        e= x + res
        if x.shape[2]==64 and x.shape[3]==64:
            info={
                "y":x,
                "w":res,
                "e":e,
            }
            torch.save(info,"/mnt/sda/zhouying/mulu/code/1.18/DMFN-master/test_results/Features_LayerNorm/UN"+str(datetime.datetime.now()) + ".pt")
        #这句作了修改。
        x = self.pmconv9(res)
        """

        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1





class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(FineGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 2, 3, 2, 1)
        # cnum*4 x 64 x 64

        self.conv5 = gen_conv(cnum * 2, cnum * 4, 3, 1, 1)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16)

        # attention branch

        self.pmconv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, activation='relu')
        
        self.contextul_attention = GlobalLocalAttention(in_dim=128)
        self.pmconv9 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.allconv11 = gen_conv(cnum * 8, cnum * 4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.allconv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum // 2, 3, 1, 1)

        self.allconv17 = gen_conv(cnum // 2, 3, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin[:,:3,:,:] * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x = self.contextul_attention(x,mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

       
        return x_stage2











# return one dimensional output indicating the probability of realness or fakeness
class Discriminator(BaseNet):
    def __init__(self, in_channels, cnum=64, is_global=True, act=F.leaky_relu):
        super(Discriminator, self).__init__()
        self.act = act
        self.embedding = None
        self.logit = None

        ch = cnum
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels, ch, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch))
        self.layers.append(nn.Conv2d(ch, ch * 2, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch * 2))
        self.layers.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch * 4))
        self.layers.append(nn.Conv2d(ch * 4, ch * 8, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch * 8))
        self.layers.append(nn.Conv2d(ch * 8, ch * 8, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch * 8))
        if is_global:
            self.layers.append(nn.Conv2d(ch * 8, ch * 8, kernel_size=5, padding=2, stride=2))
            self.layers.append(nn.BatchNorm2d(ch * 8))
            self.is_global = True
        else:
            self.is_global = False
        self.layers.append(nn.Linear(ch * 8 * 4 * 4, 512))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, middle_output=False):
        bottleneck = []
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = self.act(x)
                bottleneck += [x]
        if self.is_global:
            bottleneck = bottleneck[:-1]
        self.embedding = x.view(x.size(0), -1)
        self.logit = self.layers[-1](self.embedding)
        if middle_output:
            return bottleneck
        else:
            return self.logit


class GlobalLocalDiscriminator(BaseNet):
    def __init__(self, in_channels, cnum=32, act=F.leaky_relu):
        super(GlobalLocalDiscriminator, self).__init__()
        self.act = act

        self.global_discriminator = Discriminator(in_channels=in_channels+3, is_global=True, cnum=cnum,
                                                  act=act)
        self.local_discriminator = Discriminator(in_channels=in_channels+3, is_global=False, cnum=cnum,
                                                 act=act)
        self.condaitional_dis= Discriminator(in_channels=in_channels+3, is_global=False, cnum=cnum,
                                                 act=act)
        self.liner = nn.Linear(1024, 1)
        self.l1 = nn.L1Loss()

    def forward(self, mode, *input):
        if mode == 'dis':
            return self.forward_adv(*input)
        elif mode == 'adv':
            return self.forward_adv(*input)
        else:
            return self.forward_fm_dis(*input)

    def forward_adv(self, x_g, x_l, edge, edge_local):
        x_g_AB = torch.cat((x_g,edge),1)
        x_l_AB  = torch.cat((x_l,edge_local),1)
        x_global = self.global_discriminator(x_g_AB)
        x_local  = self.local_discriminator(x_l_AB)
        ca = torch.cat([x_global, x_local], -1)
        logit = self.liner(F.leaky_relu(ca))
        return logit

    def forward_fm_dis(self, real, fake, weight_fn, edge):
        fake_AB = torch.cat((fake,edge), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        real_AB = torch.cat((real,edge), 1)
        Dreal = self.condaitional_dis(real_AB, middle_output=True)
        Dfake = self.condaitional_dis(fake_AB, middle_output=True)
        fm_dis_list = []
        for i in range(5):
            fm_dis_list += [F.l1_loss(Dreal[i], Dfake[i], reduction='sum') * weight_fn(Dreal[i])]
        fm_dis = reduce(lambda x, y: x + y, fm_dis_list)
        return fm_dis


class InpaintingModel_DFBM(BaseModel):
    def __init__(self, act=F.elu, opt=None):
        super(InpaintingModel_DFBM, self).__init__()
        self.opt = opt
        self.init(opt)

        self.confidence_mask_layer = ConfidenceDrivenMaskLayer()

        self.netDFBN = Generator().cuda()
        init_weights(self.netDFBN)
        self.model_names = ['DFBN']
        if self.opt.phase == 'test':
            return

        self.netD = None
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netDFBN.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        self.optimizers += [self.optimizer_G]
        self.optimizer_D = None
        self.zeros = torch.zeros((opt.batch_size, 1)).cuda()
        self.ones = torch.ones((opt.batch_size, 1)).cuda()
        self.aeloss = nn.L1Loss()
        self.vggloss = None
        self.G_loss = None
        self.G_loss_mrf = None
        self.G_loss_adv, self.G_loss_vgg, self.G_loss_fm_dis = None, None, None
        self.G_loss_ae = None
        self.loss_eta = 5
        self.loss_mu = 0.03
        self.loss_vgg = 1
        self.BCEloss = nn.BCEWithLogitsLoss().cuda()
        self.gt, self.gt_local = None, None
        self.mask, self.mask_01 = None, None
        self.rect = None
        self.im_in, self.gin = None, None

        self.completed, self.completed_local = None, None
        self.completed_logit, self.gt_logit = None, None

        def weight_fn(layer):
            s = layer.shape
            return 1e3 / (s[1] * s[1] * s[1] * s[2] * s[3])

        self.weight_fn = weight_fn

        self.pred = None

        self.netD = GlobalLocalDiscriminator(3, cnum=opt.d_cnum, act=F.leaky_relu).cuda()
        init_weights(self.netD)
        self.optimizer_D = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()), lr=opt.lr,
                                            betas=(0.5, 0.9))
        self.vggloss = VGGLoss()
        self.svd_loss = svd_loss()
        self.tv_loss = TVLoss()
        self.optimizers += [self.optimizer_D]
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, [2000, 40000], 0.5))

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def update_learning_rate(self):
        for schedular in self.schedulers:
            schedular.step()
 
    def initVariables(self):
        self.gt = self.input['gt']
        self.edge = self.edge
        mask, rect = generate_mask(self.opt.mask_type, self.opt.img_shapes, self.opt.mask_shapes)
        self.mask_01 = torch.from_numpy(mask).cuda().repeat([self.opt.batch_size, 1, 1, 1])
        self.mask = self.confidence_mask_layer(self.mask_01)
        if self.opt.mask_type == 'rect':
            self.rect = [rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3]]
            self.gt_local = self.gt[:, :, self.rect[0]:self.rect[0] + self.rect[1],
                            self.rect[2]:self.rect[2] + self.rect[3]]

            self.edge_local = self.edge[:, :1, self.rect[0]:self.rect[0] + self.rect[1],
                         self.rect[2]:self.rect[2] + self.rect[3]]
        else:
            self.gt_local = self.gt
            self.edge_local = self.edge
        self.im_in = self.gt * (1 - self.mask_01)  #这个貌似是缺失的地方是1
        #        #马赛克里面貌似1是空缺，0 不缺
        self.gin = torch.cat((self.im_in, self.edge), 1)

    def Dra(self, x1, x2):
        return x1 - torch.mean(x2)





    def forward_G(self):
        # self.G_loss_reconstruction = self.recloss(self.completed * self.mask, self.gt.detach() * self.mask)
        # self.G_loss_reconstruction = self.G_loss_reconstruction / torch.mean(self.mask_01)
        self.G_loss_ae = self.aeloss(self.completed_local, self.gt_local)

        # vgg loss
        mask_error = torch.mean(F.mse_loss(self.completed_local, self.gt_local, reduction='none'), dim=1)
        mask_max = mask_error.max(1, True)[0].max(2, True)[0]
        mask_min = mask_error.min(1, True)[0].min(2, True)[0]
        mask_guidance = (mask_error - mask_min) / (mask_max - mask_min)

        self.G_loss_vgg = self.vggloss(self.completed_local, self.gt_local.detach(), mask_guidance.detach(), self.weight_fn)

        # adv loss

        xf = self.netD('adv', self.completed, self.completed_local, self.edge, self.edge_local)
        xr = self.netD('adv', self.gt, self.gt_local, self.edge, self.edge_local)
        self.G_loss_adv = (self.BCEloss(self.Dra(xr, xf), self.zeros) + self.BCEloss(self.Dra(xf, xr), self.ones)) / 2

        # fm dis loss

        self.G_loss_fm_dis = self.netD('fm_dis', self.gt_local, self.completed_local, self.weight_fn,self.edge_local)

        self.G_loss = self.G_loss_ae + self.loss_vgg * self.G_loss_vgg + self.loss_mu * self.G_loss_adv + self.loss_eta * self.G_loss_fm_dis #+\
            #self.svd_loss(self.gt_local, self.completed_local) + self.svd_loss(self.completed, self.gt)
        self.svd_loss_inter = (self.svd_loss(self.gt_local, self.completed_local) + self.svd_loss(self.completed, self.gt)) 
        self.tv = torch.abs(self.tv_loss(self.completed)- self.tv_loss( self.gt)) 
        #print(self.tv)
       

    def forward_D(self):

        xf = self.netD('dis', self.completed.detach(), self.completed_local.detach(), self.edge, self.edge_local)
        xr = self.netD('dis', self.gt, self.gt_local, self.edge, self.edge_local)

        #print("xr")
        #print(xr)
        #print("xf")
        #print(xf)
        # hinge loss
        self.D_loss = (self.BCEloss(self.Dra(xr, xf) , self.ones) + self.BCEloss(self.Dra(xf, xr) + self.svd_loss(xr, xf) , self.zeros)) / 2

    def backward_G(self):
        self.G_loss.backward()

    def backward_D(self):
        self.D_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        self.initVariables()

        _,self.pred = self.netDFBN(self.gin , self.mask_01 )
        #size=【8，3，256，256】 self.pred.requires_grad=true
        self.completed = self.pred * self.mask_01 + self.gt * (1 - self.mask_01)
        if self.opt.mask_type == 'rect':
            self.completed_local = self.completed[:, :, self.rect[0]:self.rect[0] + self.rect[1],
                                   self.rect[2]:self.rect[2] + self.rect[3]]
            self.edge_local = self.edge[:, :, self.rect[0]:self.rect[0] + self.rect[1],
                                   self.rect[2]:self.rect[2] + self.rect[3]]
        else:
            self.completed_local = self.completed

        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.forward_G()
        self.backward_G()
        self.optimizer_G.step()

        for p in self.netD.parameters():
            p.requires_grad = True

        for i in range(self.opt.D_max_iters):
            self.optimizer_D.zero_grad()
            self.forward_D()
            self.backward_D()
            self.optimizer_D.step()






    def get_current_losses(self):
        l = {'G_loss': self.G_loss.item(), 'G_loss_ae': self.G_loss_ae.item(), }
        if self.opt.pretrain_network is False:
            l.update({'G_loss_adv':        self.G_loss_adv.item(),
                      'G_loss_vgg':        self.G_loss_vgg.item(),
                      'G_loss_vgg_align':  self.vggloss.align_loss.item(),
                      'G_loss_vgg_guided': self.vggloss.guided_loss.item(),
                      'G_loss_vgg_fm':     self.vggloss.fm_vgg_loss.item(),
                      'D_loss':            self.D_loss.item(),
                      'G_loss_fm_dis':     self.G_loss_fm_dis.item(),
                      'svd':               self.svd_loss_inter.item()})
        return l

    def get_current_visuals(self):
        return {'input':     self.im_in.cpu().detach().numpy(), 'gt': self.gt.cpu().detach().numpy(),
                'completed': self.completed.cpu().detach().numpy()}

    def get_current_visuals_tensor(self):
        return {'input':     self.im_in.cpu().detach(), 'gt': self.gt.cpu().detach(),
                'completed': self.completed.cpu().detach()}

    def evaluate(self, image,edge, mask):
        image = torch.from_numpy(image).type(torch.FloatTensor).cuda() / 127.5 - 1
        mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda()
        edge = torch.from_numpy(edge).type(torch.FloatTensor).cuda()

        im_in = image * (1 - mask)
        xin = torch.cat((im_in, edge), 1)
        _, ret = self.netDFBN(xin, mask)
        ret = ret * mask + im_in * (1 - mask)
        ret = (ret.cpu().detach().numpy() + 1) * 127.5
        return ret.astype(np.uint8)




def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation, weight_norm='sn')


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='lrelu', pad_type='zeros', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
            self.pad_type = 'reflect'
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
            self.pad_type = 'replicate'
        elif pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
            self.pad_type = 'zeros'
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)

        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        # self.linear = nn.Linear(self.cnum * 4 * 8 * 8, 1)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dis_conv_module(x)
        # x = self.dropout(x)
        # x = x.view(x.size()[0], -1)
        # x = self.linear(x)

        return x

class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum * 2, 5, 2, 2)
        self.conv3 = dis_conv(cnum * 2, cnum * 4, 5, 2, 2)
        self.conv4 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2)
        self.conv5 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)


        return x
