import tensorflow as tf
import numpy as np
from vgg_preprocessing import preprocess_image
# import build_imagenet_data
# from build_imagenet_data import _process_image
# from build_imagenet_data import ImageCoder
#import matplotlib.pyplot as plt

	
def image_to_act_vgg(f_name):
	#os.system('taskset -p 0xffffffff %d' % os.getpid())
	path ="../ILSVRC2012_img_val/"    #指定需要读取文件的目录
	print(path+f_name)
	#image_raw_data = tf.read_file(path+f_name)
	image_raw_data = tf.gfile.GFile(path + f_name, 'rb').read()
	ima_data = tf.image.decode_jpeg(image_raw_data, channels=3)
	# if ima_data.dtype != tf.float32:
	# 	ima_data = tf.image.convert_image_dtype(ima_data, dtype=tf.float32)
	# ima_data = tf.image.central_crop(ima_data, central_fraction=0.875)
	# #image = tf.image.decode_jpeg(tf.read_file(file_input))
	# images = tf.expand_dims(ima_data, 0)
	# #images = tf.cast(images, tf.float32) / 256.
	# #sess=tf.Session()
	# #print(sess.run(images).max())
	# #images.set_shape((None, None, None, 3))
	# #images = tf.image.resize_images(images, (224, 224))
	# # Resize the image to the specified height and width.
	# #image = tf.expand_dims(images, 0)
	# images = tf.image.resize_bilinear(images, [224, 224],
	# 								align_corners=False)
	# images = tf.squeeze(images, [0])
	# images = tf.subtract(images, 0.5)
	# images = tf.multiply(images, 2.0)
	images = preprocess_image(ima_data, 224, 224)
	with tf.Session() as Sess:
		#print(images)
        #boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
		#result = preprocess_for_train(ima_data, 224, 224,bbox=None)
		img_numpy=Sess.run(images)
		#print(img_numpy)
		#print(img_numpy.shape)
		#img_numpy=img_numpy.reshape(224,224,3)
		img_numpy=np.moveaxis(img_numpy,-1,0)
	return img_numpy