import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats

################################################
data_folder = "../../DATA/KEY2_CABLE/CW_FORMAT/traces/"

#################################################
def Sbox_inv(input):
  Sin = np.array([
            0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3,
            0x9e, 0x81, 0xf3, 0xd7, 0xfb , 0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f,
            0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb , 0x54,
            0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b,
            0x42, 0xfa, 0xc3, 0x4e , 0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24,
            0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25 , 0x72, 0xf8,
            0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d,
            0x65, 0xb6, 0x92 , 0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
            0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84 , 0x90, 0xd8, 0xab,
            0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3,
            0x45, 0x06 , 0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1,
            0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b , 0x3a, 0x91, 0x11, 0x41,
            0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6,
            0x73 , 0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9,
            0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e , 0x47, 0xf1, 0x1a, 0x71, 0x1d,
            0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b ,
            0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0,
            0xfe, 0x78, 0xcd, 0x5a, 0xf4 , 0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07,
            0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f , 0x60,
            0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f,
            0x93, 0xc9, 0x9c, 0xef , 0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5,
            0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61 , 0x17, 0x2b,
            0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55,
            0x21, 0x0c, 0x7d
            ])
  return Sin[input]

def hamming_weight(n):
    return bin(np.uint8(n)).count("1")

def t_test(traces_0,traces_1,name):

	print(len(traces_0),len(traces_1))

	(statistic,pvalue) = stats.ttest_ind(traces_0, traces_1, equal_var = False)

	m_index = statistic.argsort()[-5:][::-1]
	tmp_p = statistic[m_index][0]
	print(m_index, 'max positive SOST =', tmp_p)

	m_index = statistic.argsort()[0:5][::]
	tmp_n = statistic[m_index][0]
	print(m_index, 'max negative SOST =', tmp_n)

	plt.ylabel('T-test score')
	plt.xlabel('Trace point') 

	plt.plot(statistic, label = name)

	return 

def run_hw_ttest_state(traces,labels,bound,name):

	labels_hw = np.vectorize(hamming_weight)(labels)

	labels_hw_state = labels_hw[:,0]
	for i in range(1,16):
		labels_hw_state = labels_hw_state+labels_hw[:,i] 

	trace_0 = traces[np.where(labels_hw_state < bound)]
	trace_1 = traces[np.where(labels_hw_state > bound)]
	t_test(trace_0,trace_1, str(name))

	return

def min_max_norm(traces):

	delimitedTraces = np.zeros(traces.shape)
	for x_index in range(traces.shape[0]):
		delimitedTraces[x_index,:] = -1+(traces[x_index,:]-np.min(traces[x_index,:]))*2/(np.max(traces[x_index,:])-np.min(traces[x_index,:]))

	return delimitedTraces

def load_traces():
		
	traces = np.load(data_folder+'2024_traces_std.npy', mmap_mode="r")[:,:] # change to 2024_traces_min_max.npy for antenna traces
	print("shape of traces",traces.shape)
	n = traces.shape[0]

	# standardization of traces, column-wise, for all trace points - NEEDED ONLY FOR ANTENNA TRACES, COMMENT OUT FOR CABLE TRACES, THEY ARE ALREADY STANDARDIZED
	#mean = np.mean(traces, axis=0)
	#std = np.std(traces, axis=0)
	#traces = (traces-mean)/std
	
	keys = np.load(data_folder+'2024_knownkey.npy', mmap_mode="r")[:n,:] 
	print("shape of keys", keys.shape)
	print('known key = ', keys[0])

	rk10 = np.load(data_folder+'rk10.npy', mmap_mode="r")[:n,:]  
	print("shape of RK10", rk10.shape)

	pt_ctr0 = np.load(data_folder+'2024_textin_ctr0.npy', mmap_mode="r")[:n,:]
	print("shape of pt_ctr0",pt_ctr0.shape)

	pt_ctr1 = np.load(data_folder+'2024_textin_ctr1.npy', mmap_mode="r")[:n,:]
	print("shape of pt_ctr1",pt_ctr1.shape)

	pt_b0 = np.load(data_folder+'2024_textin_b0.npy', mmap_mode="r")[:n,:]
	print("shape of pt_b0",pt_b0.shape)

	ct_ctr0 = np.load(data_folder+'2024_textout_ctr0.npy', mmap_mode="r")[:n,:]
	print("shape of ct_ctr0",ct_ctr0.shape)

	ct_ctr1 = np.load(data_folder+'2024_textout_ctr1.npy', mmap_mode="r")[:n,:]
	print("shape of ct_ctr1",ct_ctr1.shape)

	ct_b0 = np.load(data_folder+'2024_textout_b0.npy', mmap_mode="r")[:n,:]
	print("shape of ct_b0",ct_b0.shape)

	# Creating pt_b1 as textout_b0 xored with [0 1 0 .. 0]
	print("original ct_b0[0]:")
	print(ct_b0[0])

	# creating pt-b0 copy
	pt_b1 = np.copy(ct_b0)

	pt_b1[:,1] = pt_b1[:,1] ^ 1 

	print("resulting pt_b1[0]:")
	print(pt_b1[0])

	#np.save(data_folder+"2024_textin_b1.npy", pt_b1)
	#quit()

	ct_b1 = np.load(data_folder+'2024_textout_b1.npy', mmap_mode="r")[:n,:]
	print("shape of ct_b1",ct_b1.shape)
	
	msg = np.load(data_folder+'2024_textin_msg.npy', mmap_mode="r")[:n,:]
	print("shape of msg",msg.shape)
	
	# Creating pt_b2 as textout_b1 xored with message
	
	pt_b2 = ct_b1 ^ msg

	#np.save(data_folder+"2024_textin_b2.npy", pt_b2)
	#quit()

	ct_b2 = np.load(data_folder+'2024_textout_b2.npy', mmap_mode="r")[:n,:]
	print("shape of ct_b2",ct_b2.shape)

# cut and append 38 point traces representing 5 encryptions
#Encryption 1: [0:38],   
#Encryption 2: [88:126]  
#Encryption 3: [628:666]  
#Encryption 4: [675:713]  
#Encryption 5: [800:838]  

	traces_new = np.concatenate(
    (traces[:, 0:38], traces[:, 88:126], traces[:, 628:666], traces[:, 675:713], traces[:, 800:838]), 
    axis=0
	)

	rk10_new = np.tile(rk10, (5, 1))
	keys_new = np.tile(keys, (5, 1))
	pt_new = np.concatenate((pt_ctr0, pt_ctr1, pt_b0, pt_b1, pt_b2), axis=0)
	ct_new = np.concatenate((ct_ctr0, ct_ctr1, ct_b0, ct_b1, ct_b2), axis=0)

	print("final shape of new traces",traces_new.shape)
	print("final shape of new keys", keys_new.shape)
	print("final shape of new pt",pt_new.shape)
	print("final shape of ct",ct_new.shape)

	np.save(data_folder+"2024_cut_and_joined_180K_traces.npy", traces_new)
	np.save(data_folder+"2024_cut_and_joined_180K_knownkey.npy", keys_new)
	np.save(data_folder+"2024_cut_and_joined_180K_keylist.npy", keys_new)
	np.save(data_folder+"2024_cut_and_joined_180K_textin.npy", pt_new)
	np.save(data_folder+"2024_cut_and_joined_180K_textout.npy", ct_new)  

	print("T-test st9 HW:")
	st9 = Sbox_inv(ct_new ^ rk10_new)
	run_hw_ttest_state(traces_new,st9,64,'st9')

	plt.legend()
	plt.show()

	return 

# Start of execution
init = time.time()

# Load traces 
load_traces()

end = time.time()

print("The total running time was: ",(end-init), " seconds.") 
