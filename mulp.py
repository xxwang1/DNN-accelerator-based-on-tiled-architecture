import os
import multiprocessing as mp
import Vgg1
from Vgg1 import VGG_16

# os.system('taskset -p 0xffffffff %d' % os.getpid())
index=9
if __name__ == '__main__':
	for i in range(os.cpu_count()):
		P = mp.Process(target=VGG_16, args=(i*209+5016*index, (i+1)*209+5016*index, 8, 8, 'vgg_negact_4scale_1000ratio_2var.txt', True))
		os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		P.start()
		# if i == 0:
		# 	P = mp.Process(target=VGG_16, args=(210, 310, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 1:
		# 	P = mp.Process(target=VGG_16, args=(310, 364, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 20:
		# 	P = mp.Process(target=VGG_16, args=(364, 418, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 2:
		# 	P = mp.Process(target=VGG_16, args=(540, 640, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 3:
		# 	P = mp.Process(target=VGG_16, args=(740, 840, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 3:
		# 	P = mp.Process(target=VGG_16, args=(840, 940, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 4:
		# 	P = mp.Process(target=VGG_16, args=(940, 1040, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 5:
		# 	P = mp.Process(target=VGG_16, args=(1040, 1140, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 6:
		# 	P = mp.Process(target=VGG_16, args=(1140, 1240, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 7:
		# 	P = mp.Process(target=VGG_16, args=(1240, 1340, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 8:
		# 	P = mp.Process(target=VGG_16, args=(1340, 1400, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 21:
		# 	P = mp.Process(target=VGG_16, args=(1400, 1463, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 9:
		# 	P = mp.Process(target=VGG_16, args=(2718, 2818, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 10:
		# 	P = mp.Process(target=VGG_16, args=(2818, 2918, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 11:
		# 	P = mp.Process(target=VGG_16, args=(2918, 3018, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 12:
		# 	P = mp.Process(target=VGG_16, args=(3018, 3075, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 22:
		# 	P = mp.Process(target=VGG_16, args=(3075, 3135, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 13:
		# 	P = mp.Process(target=VGG_16, args=(3345, 3445, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 14:
		# 	P = mp.Process(target=VGG_16, args=(3445, 3498, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 23:
		# 	P = mp.Process(target=VGG_16, args=(3498, 3553, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 15:
		# 	P = mp.Process(target=VGG_16, args=(3763, 3863, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 16:
		# 	P = mp.Process(target=VGG_16, args=(3863, 3971, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 17:
		# 	P = mp.Process(target=VGG_16, args=(4390, 4490, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 18:
		# 	P = mp.Process(target=VGG_16, args=(4490, 4544, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()
		# if i == 19:
		# 	P = mp.Process(target=VGG_16, args=(4544, 4598, 8, 8, 'arbitrary_infratio_0var_8adc_vgg_perlayer.txt', True))
		# 	os.system("taskset -p -c %d %d" % (i % os.cpu_count(), os.getpid()))
		# 	P.start()