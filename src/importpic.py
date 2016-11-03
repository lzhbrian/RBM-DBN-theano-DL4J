# -*- coding:utf-8 -*-  
import cv2 
import numpy
import datetime
import sys
import os
import theano
import theano.tensor as T

from theano import pp


# 输入的格式：python importpic.py [测试集们所在目录]

# Output info & time
print '\nWork by Lin, Tzu-Heng'
print '2014011054, W42, Dept. of Electronic Engineering, Tsinghua University'
print 'All rights reserved\n'
starttime = datetime.datetime.now()
print ('Started at '+starttime.strftime("%Y-%m-%d %H:%M:%S"))

# 图片文件夹路径(file dir)
dire = sys.argv[1]
if dire[len(dire)-1]!='/':
	dire = dire + '/'

# 遍历目录下的所有图片
# # 重命名
# for item in os.listdir(dire):
# 	os.chdir(dire)
# 	os.rename(item, item[4:7]+'.png')
# 	# 如果是文件且后缀名是.png

# # 截取图片成28x28	
# for item in os.listdir(dire):
# 	if item.endswith('.png'):
# 		# Read in the files
# 		img = cv2.imread(dire+item) 
# 		print str(img)
# 		img2 = img[2:30,2:30]
# 		cv2.imwrite(item[0:3]+'.png',img2)
# 		print 'Written'

# 遍历目录下的所有图片

label = [] # 标签数组
flag = 0
# target 存测试集矩阵
for item in os.listdir(dire):
	# 如果是文件且后缀名是.png
	if item.endswith('.png'):
		# Read in the files
		img = cv2.imread(dire+item) 
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换为grayscle
		img = img.reshape(1,784) # 变成1x784的矩阵
		if flag == 0:
			target = 255 - img
			flag = 1
		else:
			target = numpy.row_stack((target, 255 - img))
		label.append(int(item[0]))


# 写进文件测试看看是否正确
fp = open('/Users/Brian/Desktop/test.txt','w')
print >> fp, label
for t in target:
	print >> fp, str(t)


# share, readable by theano
test_X_matrix = numpy.asarray(target,dtype = float)
test_set_x = theano.shared(test_X_matrix) #

test_Y_vector = numpy.asarray(label)
shared_var_Y = theano.shared(test_Y_vector)
test_set_y = T.cast(shared_var_Y, 'int32') #

print >> fp, test_set_x.type
print >> fp, test_set_y.type






