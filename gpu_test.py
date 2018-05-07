import numpy as np
import os
while 1:
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
	memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
	os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
	os.system('rm tmp')
