import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
def distort_color(image,color_ordering=0):
	if color_ordering == 0:
	    image = tf.image.random_brightness(image,max_delta=32./255.)#亮度
	    image = tf.image.random_saturation(image,lower=0.5,upper=1.5)#饱和度 
	    image = tf.image.random_hue(image,max_delta=0.2)#色相 
	    image = tf.image.random_contrast(image,lower=0.5,upper=1.5)#对比度 
	elif color_ordering == 1: 
	    image = tf.image.random_brightness(image, max_delta=32. / 255.) # 亮度 
	    image = tf.image.random_hue(image, max_delta=0.2) # 色相 
	    image = tf.image.random_saturation(image, lower=0.5, upper=1.5) # 饱和度 
	    image = tf.image.random_contrast(image, lower=0.5, upper=1.5) # 对比度 
	return tf.clip_by_value(image,0.0,1.0) #将张量值剪切到指定的最小值和最大值
def preprocess_for_train(image,height,width,bbox):
#如果没有提供标注框，则认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])

    #转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)

    #随机截取图像，减少需要关注的物体大小对图像识别的影响
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distort_image = tf.slice(image,bbox_begin,bbox_size)

    #将随机截图的图像调整为神经网络输入层的大小。大小调整的算法是随机的
    distort_image = tf.image.resize_images( distort_image,[height,width],method=np.random.randint(4) )

    #随机左右翻转图像
    distort_image = tf.image.random_flip_left_right(distort_image)

    #使用一种随机的顺序调整图像色彩
    distort_image = distort_color(distort_image,np.random.randint(1))
    return distort_image
def image_to_act():
	image_raw_data = tf.gfile.FastGFile("ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG",'rb').read()
	with tf.Session() as Sess:
		ima_data = tf.image.decode_jpeg(image_raw_data)
		boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
		result = preprocess_for_train(ima_data, 224, 224, boxes)
		img_numpy=result.eval(session=Sess)
		img_numpy=np.moveaxis(img_numpy,-1,0)
	return img_numpy
