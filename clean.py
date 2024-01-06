import matplotlib.pyplot  as plt
import pandas as pd
import numpy as np
import os
from PIL import Image



info_record = pd.read_excel("~/dataset/puzzle2k/my0424.xlsx", sheet_name="Sheet1", header = 0)
print(".columns",info_record.columns)#Sheet1
print(info_record.index)
'''
def getpicname(data_info):
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
    return name_sp
name_sp = getpicname(info_record)#0424
file_path = "/homeB/liangxiaoyu/dataset/puzzle2k/img/"
for i in range(10219):
    img_fn = os.path.join(file_path, name_sp[i])
    try:
        single_img_img = Image.open(img_fn).verify()
    except:
        print(img_fn)
'''
# xdata = info_record.index[:5000] #横坐标 时间是列名
# # y1data = data.loc[:, '列名1'] #多条曲线的y值 参数名为csv的列名
# y2data = info_record.values[:,3][:5000]
# y3data = info_record.values[:,2][:5000]
# plt.plot(xdata, y2data, color='b', label=u'2路')#星形标记,蓝色
# plt.plot(xdata,4*(y3data), color='g', label=u'3路')#上三角标记,绿色
# plt.plot(xdata,(y3data), color='r', label=u'1路')#上三角标记,绿色
# plt.savefig("/homeB/liangxiaoyu/23w0322/puzzle/output_dir/result/try.jpg")
# plt.close()


print("i")
file1 = np.array(info_record)
data = []
dirtydata = []
for item in file1:
    t6 = float(item[2])
    t10 = float(item[3])
    if (t10 > 4*t6 )or (t10<1.1*t6):#(t10 > 4*t6 )or (t10<1.1*t6)
        dirtydata.append(item)
    else:
        data.append(item)


np.savetxt("0424cleaned.csv",data,delimiter=",", fmt = '%s')
np.savetxt("0424becleaneddirty.csv",dirtydata,delimiter=",", fmt = '%s')

