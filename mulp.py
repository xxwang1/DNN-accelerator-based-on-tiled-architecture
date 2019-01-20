import main
from main import MobileNet
import multiprocessing as mp
if __name__ == '__main__':
	P1 = mp.Process(target=MobileNet, args=(23,6250))
	P2 = mp.Process(target=MobileNet, args=(6262,12500))
	P3 = mp.Process(target=MobileNet, args=(12507,18750))
	P4 = mp.Process(target=MobileNet, args=(18799,25000))
	P5 = mp.Process(target=MobileNet, args=(25015,31250))
	P6 = mp.Process(target=MobileNet, args=(31261,37500))
	P7 = mp.Process(target=MobileNet, args=(37508,43750))
	P8 = mp.Process(target=MobileNet, args=(43755,50000))
	P1.start()
	P2.start()
	P3.start()
	P4.start()
	P5.start()
	P6.start()
	P7.start()
	P8.start()

	P1.join()
	P2.join()
	P3.join()
	P4.join()
	P5.join()
	P6.join()
	P7.join()
	P8.join()
                           