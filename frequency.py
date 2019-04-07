import glob
import data_process
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import medfilt
import math
from tqdm import tqdm
import pandas as pd
import pickle
import scipy.fftpack
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean,sqeuclidean
from fastdtw import fastdtw
from dtw import dtw
from scipy import signal
import re
from numpy.fft import fft, fftfreq, ifft

def process(txt_fileloc,graph,fs,lowcut,highcut):

	import csv 
	import math
	import re 
	from mpl_toolkits import mplot3d
	import matplotlib.pyplot as plt
	from scipy.signal import butter, lfilter
	from scipy import signal

	filename = txt_fileloc
	csv_fileloc_1 = filename.replace(".txt", ".csv")
	csv_fileloc = csv_fileloc_1.replace("/Data", "/Data_csv")
	file_final = []

	#extracting data from text file

	f = open(txt_fileloc,'r')
	file_contents = f.read().split('\n')[:-1]
	for row in file_contents:
		file_final.append(row.split(' '))
	f.close()

	x_final = [item[0] for item in file_final]
	y_final = [item[1] for item in file_final]
	z_final = [item[2] for item in file_final]
	a_final = [math.sqrt((int(item[0]))**2 + (int(item[1]))**2 + (int(item[2]))**2) for item in file_final]
	# converting co-ordinates to integer

	x_final= list(map(int,x_final))
	y_final= list(map(int,y_final))
	z_final= list(map(int,z_final))


	m = re.search('/Users/Liuzhaoyu/Desktop/research/WHARF_Dataset/WHARF-master/WHARF-master/WHARF-Data-Set/Data/(.+?)/.*', filename)
	return x_final,y_final,z_final,a_final,m.group(1)

file_name = []
x_mean = []
y_mean = []
z_mean = []
x_std = []
y_std = []
z_std = []
xy_corr = []
xz_corr = []
yz_corr = []
x_energy = []
y_energy = []
z_energy = []
pitch    = []
roll     = []
yaw      = []
a_label = []
s_label =[]
e_label = []
total_x = []
total_y = []
total_z = []
total_a = []
label_a = []
label_s = []
label_e = []
segment_x = []
segment_y = []
segment_z = []
segment_a = []

act = {'Brush_teeth':0,'Climb_stairs':1,'Comb_hair':2,'Descend_stairs':3,'Drink_glass':4,'Eat_meat':5,'Eat_soup':6,'Getup_bed':7,'Liedown_bed':8,'Pour_water':9,'Sitdown_chair':10,'Standup_chair':11,'Use_telephone':12,'Walk':13}
type_act = {'Brush_teeth':0,'Climb_stairs':0,'Comb_hair':0,'Descend_stairs':1,'Drink_glass':0,'Eat_meat':0,'Eat_soup':0,'Getup_bed':2,'Liedown_bed':2,'Pour_water':0,'Sitdown_chair':2,'Standup_chair':2,'Use_telephone':0,'Walk':1}
eat_act = {'Brush_teeth':0,'Climb_stairs':0,'Comb_hair':0,'Descend_stairs':0,'Drink_glass':0,'Eat_meat':1,'Eat_soup':1,'Getup_bed':0,'Liedown_bed':0,'Pour_water':0,'Sitdown_chair':0,'Standup_chair':0,'Use_telephone':0,'Walk':0}

for name in glob.glob('/Users/Liuzhaoyu/Desktop/research/WHARF_Dataset/WHARF-master/WHARF-master/WHARF-Data-Set/Data/*/*.txt'):
	file_name.append(name)

lengths = []

name1 = re.search('/Users/Liuzhaoyu/Desktop/research/WHARF_Dataset/WHARF-master/WHARF-master/WHARF-Data-Set/Data/(.+?)/.*', file_name[0])

previous = [act[name1.group(1)]]
change = []
change_copy = []
change_label = []
for i in tqdm(file_name):
	x,y,z,a,name = process(i,0,32,1.6,15)
	lengths.append(len(x))
	total_x.extend(x)
	total_y.extend(y)
	total_z.extend(z)
	total_a.extend(a)
	label_a += len(x) * [act[name]]
	label_s += len(x) * [type_act[name]]
	label_e += len(x) * [eat_act[name]]
	current = [act[name]]
	if (previous == current):
		previous = current
	else:
		previous = current
		change.append(len(total_x))
		change_copy.append(len(total_x))
		change_label.append([act[name]])

# 10 sec
window_size = 320

## 5 sec data 
#window_size = 160 #sec *fs
overlap = int(window_size*1.0)

tot_labe = []
for i in range(0, len(total_x),overlap):

	if i+window_size <= len(total_x):
		x1 = [total_x[i:i+window_size]]
		y1 = [total_y[i:i+window_size]]
		z1 = [total_z[i:i+window_size]]
		a1 = [total_a[i:i+window_size]]

		segment_x.extend(x1)
		segment_y.extend(y1)
		segment_z.extend(z1)
		segment_a.extend(a1)
		label_i = label_a[i:i+window_size]

		a_label.append((np.bincount(label_a[i:i+window_size]).argmax()))
		s_label.append((np.bincount(label_s[i:i+window_size]).argmax()))
		e_label.append((np.bincount(label_e[i:i+window_size]).argmax()))
		tot_label_in = 32*[(np.bincount(label_a[i:i+window_size]).argmax())]
		tot_labe.extend(tot_label_in)

label_compare = label_a[0:i]

segment_x = np.array([np.array(xi) for xi in segment_x])
segment_y = np.array([np.array(yi) for yi in segment_y])
segment_z = np.array([np.array(zi) for zi in segment_z])
segment_a = np.array([np.array(ai) for ai in segment_a])

# brush teeth
act_0_from = label_a.index(0)
act_0_to = len(label_a) - 1 - label_a[::-1].index(0)

# climb stairs
act_1_from = label_a.index(1)
act_1_to = len(label_a) - 1 - label_a[::-1].index(1)

# comb hair
act_2_from = label_a.index(2)
act_2_to = len(label_a) - 1 - label_a[::-1].index(2)

# decend stairs
act_3_from = label_a.index(3)
act_3_to = len(label_a) - 1 - label_a[::-1].index(3)

# drink glass
act_4_from = label_a.index(4)
act_4_to = len(label_a) - 1 - label_a[::-1].index(4)

# eat soup
act_6_from = label_a.index(6)
act_6_to = len(label_a) - 1 - label_a[::-1].index(6)

# use telephone
act_12_from = label_a.index(12)
act_12_to = len(label_a) - 1 - label_a[::-1].index(12)

# walk
act_13_from = label_a.index(13)
act_13_to = len(label_a) - 1 - label_a[::-1].index(13)


# brush teeth
total_0_x = total_x[act_0_from:act_0_to]
total_0_y = total_y[act_0_from:act_0_to]
total_0_z = total_z[act_0_from:act_0_to]
fig = plt.figure()
plt.plot(total_0_x)
fig.suptitle('brush teeth')
plt.show()

# climb stairs
total_1_x = total_x[act_1_from:act_1_to]
total_1_y = total_y[act_1_from:act_1_to]
total_1_z = total_z[act_1_from:act_1_to]
fig = plt.figure()
plt.plot(total_1_x)
fig.suptitle('climb stairs')
plt.show()

# comb hair
total_2_x = total_x[act_2_from:act_2_to]
total_2_y = total_y[act_2_from:act_2_to]
total_2_z = total_z[act_2_from:act_2_to]
fig = plt.figure()
plt.plot(total_2_x)
fig.suptitle('comb hair')
plt.show()

# decend stairs
total_3_x = total_x[act_3_from:act_3_to]
total_3_y = total_y[act_3_from:act_3_to]
total_3_z = total_z[act_3_from:act_3_to]
fig = plt.figure()
plt.plot(total_3_x)
fig.suptitle('decend stairs')
plt.show()

# drink glass
total_4_x = total_x[act_4_from:act_4_to]
total_4_y = total_y[act_4_from:act_4_to]
total_4_z = total_z[act_4_from:act_4_to]
fig = plt.figure()
plt.plot(total_4_x)
fig.suptitle('drink glass')
plt.show()

# eat soup
total_6_x = total_x[act_6_from:act_6_to]
total_6_y = total_y[act_6_from:act_6_to]
total_6_z = total_z[act_6_from:act_6_to]
fig = plt.figure()
plt.plot(total_6_x)
fig.suptitle('eat soup')
plt.show()

# use telephone
total_12_x = total_x[act_12_from:act_12_to]
total_12_y = total_y[act_12_from:act_12_to]
total_12_z = total_z[act_12_from:act_12_to]
fig = plt.figure()
plt.plot(total_12_x)
fig.suptitle('use telephone')
plt.show()

# walk
total_13_x = total_x[act_13_from:act_13_to]
total_13_y = total_y[act_13_from:act_13_to]
total_13_z = total_z[act_13_from:act_13_to]
fig = plt.figure()
plt.plot(total_13_x)
fig.suptitle('walk')
plt.show()

# climb stairs
# frequency: 1.5347332850591358
n = len(total_1_x)
act_list = total_1_x
freqs = fftfreq(n, 1.0/32)
mask = freqs > 1
fft_value = fft(act_list)
fft_theo = np.abs(fft_value/n)**2
fft_list = list(fft_theo[mask])
print freqs[mask][fft_list.index(max(fft_list))]

fig = plt.figure()
plt.plot(freqs[mask], fft_theo[mask])
fig.suptitle('climb stairs')
plt.show()

# decend stairs
# frequency: 1.847172008088731
n = len(total_3_x)
act_list = total_3_x
freqs = fftfreq(n, 1.0/32)
mask = freqs > 1
fft_value = fft(act_list)
fft_theo = np.abs(fft_value/n)**2
fft_list = list(fft_theo[mask])
print freqs[mask][fft_list.index(max(fft_list))]

fig = plt.figure()
plt.plot(freqs[mask], fft_theo[mask])
fig.suptitle('decend stairs')
plt.show()

# decend stairs
# frequency: 1.749970190671306
n = len(total_13_x)
act_list = total_13_x
freqs = fftfreq(n, 1.0/32)
mask = freqs > 1
fft_value = fft(act_list)
fft_theo = np.abs(fft_value/n)**2
fft_list = list(fft_theo[mask])
print freqs[mask][fft_list.index(max(fft_list))]

fig = plt.figure()
plt.plot(freqs[mask], fft_theo[mask])
fig.suptitle('walk')
plt.show()

