import os, sys, shutil, time, random
import tqdm

import numpy as np
import PIL.Image
import torch
import pandas as pd

import yaml
import timm
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict

from PIL.ImageFile import ImageFile
#
from timm.data import Mixup
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
#my
from datasets import build_dataset
#from samplers import RASampler
from torch.optim.lr_scheduler import LambdaLR
ImageFile.LOAD_TRUNCATED_IMAGES = True

# import models





def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


# ---------------------------------------------------------------------------- #
#yaml 配置文件
def get_yaml_data(yaml_file):
    # 打开yaml文件
    print("***获取yaml文件数据***")
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    
    print(file_data)
    print("类型：", type(file_data))

    # 将字符串转化为字典或列表
    print("***转化yaml数据为字典或列表***")
    data = yaml.load(file_data)
    print(data)
    print("类型：", type(data))
    return data
def backup_yaml(yaml_file):
    py_object = {'school': 'zhang',
                 'students': ['a', 'b']}
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(py_object, file)
    file.close()  
# ---------------------------------------------------------------------------- #
def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def save_weights(model,optimizer,epoch,filename,args):
    # save weights
    #filename = "/result/vitb_lre-4_224e30.pth"
    filepath = args.my_output_dir+"/result/"+filename+".pth"
    #print(filepath)
    save_files = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_val': 0.1}
    torch.save(save_files,filepath)

def load_weights(model, checkpoint_path ):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print('Restoring model state from checkpoint...')
            #print( checkpoint )
            model.load_state_dict(checkpoint['model'],strict=False)


def main(args):
    print("printinfo,各种信息，我记得有个函数，该输出版本的")
    print(args)
    torch.backends.cudnn.enabled = False

    #随机seed42

    #基础参数再设置与分布式
    print("分布式设置 空缺")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    # ---------------------------------------------------------------------------- #
    # 加载模型参数,做了模型融合，系数可以在infersimple（）中调整
    # ---------------------------------------------------------------------------- #
    print(f"Creating model: {args.b_model}")
    modelbase10 = timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=2)
    modelbase10.blocks = torch.nn.Sequential( *( list(modelbase10.blocks)[0:3] ) )
    modelbase10.to(device)
    print("re!")
    checkpoint_path = "./output_dir/result/0428a10basev8.pth"
    load_weights(modelbase10, checkpoint_path)
    print("sume!",checkpoint_path)
    modelbase10.to(device)

    modelsmall10 = timm.create_model("vit_small_patch16_224",pretrained=False,num_classes=2)
    modelsmall10.blocks = torch.nn.Sequential( *( list(modelsmall10.blocks)[0:3] ) )
    #modelsmall10.blocks[0].mlp.drop = torch.nn.Dropout(p=0.3, inplace=False)
    modelsmall10.to(device)
    print("re!")
    checkpoint_path = "./output_dir/result/0428a10smallv8.pth"
    load_weights(modelsmall10, checkpoint_path)
    print("sume!",checkpoint_path)
    modelsmall10.to(device)

    modelbase6 = timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=2)
    modelbase6.blocks = torch.nn.Sequential( *( list(modelbase6.blocks)[0:3] ) )
    modelbase6.to(device)
    print("re!")
    checkpoint_path = "./output_dir/result/0428a6basev6.pth"
    load_weights(modelbase6, checkpoint_path)
    print("sume!",checkpoint_path)
    modelbase6.to(device)

    modelsmall6 = timm.create_model("vit_small_patch16_224",pretrained=False,num_classes=2)
    modelsmall6.blocks = torch.nn.Sequential( *( list(modelsmall6.blocks)[0:3] ) )
    modelsmall6.to(device)
    print("re!")
    checkpoint_path = "./output_dir/result/0428a6smallv6.pth"
    load_weights(modelsmall6, checkpoint_path)
    print("sume!",checkpoint_path)        
    modelsmall6.to(device)
    print("finish_load_to_infer")
    # ---------------------------------------------------------------------------- #
    # 图片推理
    # 路径直接写在下面就好file_path，single_img_name
    # ---------------------------------------------------------------------------- #

    ## infer dataset
    ## infer dataset
    filepath = 'E:\YYT\pythonProject\code0428/0424cleaned.csv'
    print(filepath)
    data_info = pd.read_csv(filepath)
    print(".columns",data_info.columns)
    print(data_info.index)

    m = data_info.values[:,1]
    c = m.astype(str)
    print("more")
    #name_sp = np.char.split(c,sep = "/").tolist()
    name_sp = np.char.split(c,sep = "/")
    #print(name_sp)
    for i in range(len(name_sp)):
        name_sp[i] = name_sp[i][-1]
    #print(len(name_sp))
    m64 = data_info.values[:,2]
    #print(len(m64))
    m100 = data_info.values[:,3]
    #print(len(m100))
    print("name_sp,m64,m100",name_sp,m64,m100)
    loss6 =[]
    loss10 =[]

    for i,name in enumerate(name_sp):
        single_img_name,a6,a10 = infersimple(modelbase10,modelsmall10,modelbase6,modelsmall6,device,file_path=args.my_file_path, single_img_name=name)
        print("a6,a10", single_img_name, a6, a10)
        # print(abs(a6-m64[i]))
        # print(abs(a10-m100[i]))
        loss6.append(abs(a6-m64[i]))
        loss10.append(abs(a10-m100[i]))


    print(loss6)
    print(loss10)
    lenth = len(loss6)
    print("avgloss",sum(loss6)/lenth,sum(loss10)/lenth)
    exit(0)





def build_transform():
    t = []
    t.append(transforms.Resize(224))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def infersimple(model1,model2,model3,model4,device,file_path, single_img_name):

    img_fn = os.path.join(file_path, single_img_name)
    single_img_img = PIL.Image.open(img_fn).convert('RGB')
    mytransform = build_transform()
    single_img_tensor = mytransform(single_img_img).unsqueeze(0)
    single_img_tensor = single_img_tensor.to(device)
    output1 = model1(single_img_tensor)
    output2 = model2(single_img_tensor)
    output3 = model3(single_img_tensor)
    output4 = model4(single_img_tensor)
    output1 = output1.detach().cpu().numpy()
    output2 = output2.detach().cpu().numpy()
    output3 = output3.detach().cpu().numpy()
    output4 = output4.detach().cpu().numpy()
    # print(output1,output2,output3,output4)
    # exit(0)
    
    a10 = (0.5*output1[0][1]+0.5*output2[0][1])
    a6 = (0.6*output3[0][0] + 0.2*output4[0][0]+0.2*(a10-5.21)/1.39)
    return single_img_name,a6,a10

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def get_parent_args_parser():
    parser = argparse.ArgumentParser('base training and evaluation script', add_help=False)
    parser.add_argument('--b-batch-size', default=64, type=int)
    parser.add_argument('--b-epochs', default=300, type=int)
    # Model parameters
    parser.add_argument('--b-model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--b-input-size', default=224, type=int, help='images input size')
    parser.add_argument('--b-num-class', default=10, type=int, help='images input size')

    parser.add_argument('--b-eval', action='store_true', default=False, help='Perform evaluation only')
    parser.add_argument('--b-dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    # * Finetuning params
    parser.add_argument('--b-finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--b-pretrained_path', default='exps/deit_small/checkpoint.pth', type=str)
    parser.add_argument('--b-resume', default='', help='resume from checkpoint')
    parser.add_argument('--b-start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Dataset parameters
    parser.add_argument('--b-data-path', default='D:/NewDesktop/临时处理/cifar10', type=str,
                        help='dataset path')
    parser.add_argument('--b-data-set', default='CIFAR10', choices=['CIFAR10','CIFAR100', 'IMNET', 'INAT', 'INAT19', 'IMNET100'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--b-inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--b-device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--b-seed', default=0, type=int)
    parser.add_argument('--b-num_workers', default=10, type=int)
    parser.add_argument('--b-pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--b-no-pin-mem', action='store_false', dest='b_pin_mem',
                        help='')
    parser.set_defaults(b_pin_mem=True)
    # distributed training parameters
    parser.add_argument('--b-world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--b-dist_url', default='env://', help='url used to set up distributed training')
    return parser

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.fconfig_file:
        with open(args_config.fconfig_file, 'r',encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False, allow_unicode=True)
    return args, args_text

if __name__ == "__main__":
    import argparse
    #传入yaml文件中的参数，配合自定义的_parse_args函数食用
    config_parser =  argparse.ArgumentParser(description='Training Config', epilog="1",add_help=False)
    config_parser.add_argument('-fc', '--fconfig-file', default='./configs/1.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    # 存储一些不变化的参数，与项目性能无关的参数。
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet Training', epilog="2",parents=[get_parent_args_parser()])

    # 下面是添加自己个性化参数，每次训练结果保存主文件夹
    parser.add_argument('--my_output_dir', default='./output_dir')
    parser.add_argument('--my_output_filename', default='vitb_lre-4_224e30')
    #parser.print_help()

    args, args_text = _parse_args()

    # 检查保存权重文件夹是否存在，不存在则创建
    if args.my_output_dir:  
        Path(args.my_output_dir).mkdir(parents=True, exist_ok=True)  

    exp_name = 'myconfig'
    output_dir = get_outdir(args.my_output_dir if args.my_output_dir else './output_dir', exp_name)
    with open(os.path.join(output_dir,args.my_output_filename+'.yaml'), 'w', encoding='utf-8') as f:
        f.write(args_text)


    main(args)


