

# requirements

einops==0.6.0
numpy==1.23.5
pandas==1.3.1
path==16.6.0
path.py==12.5.0
pathtools==0.1.2
Pillow==9.4.0
PyYAML 
timm==0.3.2
tqdm==4.64.1

和torch框架

# 文件架构

configs放置 yaml文件，使用时修改其参数

output_dir放置训练结果pth权重文件、配置文件备份

utils.py放置一些常用的工具函数

engine.py是训练相关的函数

datasets.py划分训练集和测试集

train.py是训练代码

main.py加载模型权重，进行预测



argparse的参数只存储一些与该实验性能无关的一些参数

包括配置文件路径，主要是结果保存路径等等 进入main函数之后 只在最开始使用 

之后用配置文件yaml的参数 进行set_defaults()重载

args的参数贯穿始终 并且会进行与结果一起备份

# useage

直接运行infer.py进行推理，得到预测的64块和100块所用时间

路径直接写在下面就好file_path为文件夹路径，single_img_name为要预测图片的名称，可以在configs文件夹下的1.yaml中定义，也可自行更改
