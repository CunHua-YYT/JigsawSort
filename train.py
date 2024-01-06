import os, sys, shutil, time, random
import tqdm

import numpy as np
import PIL.Image
import torch
import torchvision
import pandas as pd

import yaml
import timm
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict
#
from timm.data import Mixup

#my
from datasets import build_dataset
#from samplers import RASampler
from engine import train_one_epoch, validate2,shortinfer
from torch.optim.lr_scheduler import LambdaLR


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
            model.load_state_dict(checkpoint['model'],strict=False)


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
    #来自tranmix
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                print('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    print('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    print('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                print("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def main(args):
    print("printinfo")
    print(args)
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # ---------------------------------------------------------------------------- #
    # data_transform, dataset, dataloader
    # ---------------------------------------------------------------------------- #
    print("Loading data...")
    #生成数据集
    dataset_train, args.b_num_class = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    print("lenth")

    # 下面是数据加载
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.b_batch_size,
        pin_memory=args.b_pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=int(args.b_batch_size),
        pin_memory=args.b_pin_mem,
        drop_last=True
    )
    # ---------------------------------------------------------------------------- #
    # 1. create model
    # 2. resume
    # 3.  optimizer+lr_scheduler
    # ---------------------------------------------------------------------------- #
    print(f"Creating model: {args.b_model}")
    model = timm.create_model("vit_small_patch16_224",pretrained=True,num_classes=2) 
    #获取timm文件夹下的vit模型，并修改最后一层的输出类别为2   ，代表对64块和100块拼图所用时间的预测
    model.blocks = torch.nn.Sequential( *( list(model.blocks)[0:3] ) )#只取前三个block


    #args.b_resume  是pth路径，加载模型权重
    if args.b_resume:
        print("re!")
        checkpoint_path = args.b_resume
        load_weights(model, checkpoint_path)
        print("sume!",checkpoint_path)


    #为了防止训练的时候过拟合，对模型的全连接层进行dropout
    model.blocks[0].mlp.drop = torch.nn.Dropout(p=0.3, inplace=False)
    #model.blocks[2].mlp.drop = torch.nn.Dropout(p=0.3, inplace=False)
    model.to(device)        

    # 优化器
    criterion = torch.nn.MSELoss(reduction='none').to(device)#在训练过程容易受离群点影响过大，L1Loss(mae)+动态lr可以很好防止
    ##给不同层分配不同学习率
    optimizer = torch.optim.SGD([{'params':model.blocks[0].parameters(),'lr':1e-4},
                        {'params':model.blocks[1].parameters(),'lr':1e-4},
                        {'params':model.blocks[2].parameters(),'lr':1e-4},
                        {'params':model.head.parameters(),'lr':1e-3}],weight_decay=1e-5)
    # 选用动态学习率，前10个epoch学习率为1，之后每个epoch学习率乘以0.95
    lr_lambda = lambda epoch:1.0 if epoch<10 else np.math.exp(0.05*(10-epoch))
    scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lr_lambda)
  

    # ---------------------------------------------------------------------------- #
    # train loop
    #     train an epoch loop(it's a function)
    #     eval(it's a function)
    #     save best model
    # ---------------------------------------------------------------------------- #
    print(f"Start training for {args.b_epochs} epochs")
    for epoch in range(args.b_epochs):
        train_one_epoch(model, data_loader_train, optimizer, criterion, device,args)
        scheduler.step()
        print("epoch+lr",epoch,"+",optimizer.param_groups[0]['lr'])
        print("finishtrain")
        validate2(model,data_loader_val,criterion,device)
        print("finishval")
        save_weights(model,optimizer,0,args.my_output_filename,args)


    print("finish_main_send_e-mile")# 提醒完成训练
    dataset_infer, _ = build_dataset(is_train=False, args=args,infer=True)
    infer_loader = torch.utils.data.DataLoader(
        dataset_infer, 
        batch_size=int(args.b_batch_size),
        pin_memory=args.b_pin_mem,
        drop_last=True
    )


    lossesavg,imgname,t64,t100,a64,a100 = shortinfer(model,infer_loader,  criterion,device)
    print("lossesavg",lossesavg)
    columns_name = ['imgname', 't6', 't10', 'a6', 'a10']
    info_record = pd.DataFrame(columns = columns_name, data = zip(imgname,t64,t100,a64,a100))
    print(info_record)
    pathh = args.my_output_filename+".csv"
    info_record.to_csv(pathh,index=None)
    l = range(len(imgname))

    draw = 0#如果需要的话，可以画图，路径在下面直接改就行
    if draw == 1:
        #data = pd.read_csv('/homeB/liangxiaoyu/23w0322/puzzle/0330toclean.csv')
        ##xdata = info_record.loc[:, '时间'] #横坐标 时间是列名
        xdata = info_record.index #横坐标 时间是列名
        # y1data = data.loc[:, '列名1'] #多条曲线的y值 参数名为csv的列名
        y2data = info_record.loc[:, 't6']
        y3data = info_record.loc[:, 'a6']

        plt.plot(xdata, y2data, color='b', label=u'2路')#星形标记,蓝色
        plt.plot(xdata, y3data, color='g', label=u'3路')#上三角标记,绿色
        plt.savefig("/homeB/xxy/23w0322/puzzle/output_dir/result/xx.jpg")
        plt.close()




    pass

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


