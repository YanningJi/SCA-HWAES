import numpy as np
import time

################################################
data_folder = "../../DATA/KEY2_CABLE/RAW_DATA/"
data_folder_cw = "../../DATA/KEY2_CABLE/CW_FORMAT/traces/"

N = 36 # number of nonce files, needs to be changes for other keys
#################################################
def min_max_norm(traces):

	delimitedTraces = np.zeros(traces.shape)
	for x_index in range(traces.shape[0]):
		delimitedTraces[x_index,:] = -1+(traces[x_index,:]-np.min(traces[x_index,:]))*2/(np.max(traces[x_index,:])-np.min(traces[x_index,:]))

	return delimitedTraces

def load_traces():
	
# Load pt and create b2 plaintext
	
	pt = np.load(data_folder+'pt_0.npy', mmap_mode="r")[:,:] 

	for i in range(1,N): 
		pt_tmp = np.load(data_folder+'pt_'+str(i)+'.npy', mmap_mode="r")[:,:] 
		pt = np.append(pt,pt_tmp,axis=0)

	print("shape of pt final",pt.shape)

	np.save(data_folder_cw+"2024_textin_msg.npy", pt)
	print("saved 2024_textin_msg.npy")
	
# Creating Counter 0
# Load nonces and create Ctr0 plaintext

	nonce = np.load(data_folder+'nonce_0.npy', mmap_mode="r")[:,:] 

	for i in range(1,N):  
		nonce_tmp = np.load(data_folder+'nonce_'+str(i)+'.npy', mmap_mode="r")[:,:] 
		nonce = np.append(nonce,nonce_tmp,axis=0)

	print("shape of nonce final",nonce.shape)
	n = nonce.shape[0]

	pt_ctr0 = np.zeros((nonce.shape[0], 16), dtype=int)

	pt_ctr0[:,0] = 1
	pt_ctr0[:,15] = 0

	for i in range(8):
		pt_ctr0[:,6+i] = nonce[:,i]

	print('Ctr-0:')
	print(pt_ctr0[0])
	print(pt_ctr0[1])
	print("shape of pt_ctr0",pt_ctr0.shape)

	np.save(data_folder_cw+"2024_textin_ctr0.npy", pt_ctr0)
	print("saved 2024_textin0.npy")

# Creating Counter 1

	pt_ctr0[:,15] = 1 # changing 0 to 1 for ctr-1

	print('Ctr-1:')
	print(pt_ctr0[0])
	print(pt_ctr0[1])
	print("shape of pt_ctr1",pt_ctr0.shape)

	np.save(data_folder_cw+"2024_textin_ctr1.npy", pt_ctr0)
	print("saved 2024_textin1.npy")

# Creating B0

	pt_ctr0[:,0] = 73
	pt_ctr0[:,15] = 16

	print('B0:')
	print(pt_ctr0[0])
	print(pt_ctr0[1])
	print("shape of b0",pt_ctr0.shape)

	np.save(data_folder_cw+"2024_textin_b0.npy", pt_ctr0)
	print("saved 2024_textin_b0.npy")
	
# Load key, expand, and save 

	keys = np.load(data_folder+'key_2.npy', mmap_mode="r")[:,:]  # update for other keys
	keys_new = np.tile(keys, (N*1000, 1))
	keys_new = keys_new[:n,:]

	print("final shape of keys ",keys_new.shape)
	print('Key:')
	print(keys_new[0])
	print(keys_new[1])

	np.save(data_folder_cw+"2024_knownkey.npy", keys_new)
	np.save(data_folder_cw+"2024_keylist.npy", keys_new)
	print("saved 2024_knownkey.npy")
	print("saved 2024_keylist.npy")

# Load all traces and save in one file 
	
	# correction to compencate for the horisontal shift in different devices
	#c = 0 		# for key 0
	#c = 85 	# for key 1  
	c = 169 	# for key 2
	
	traces = np.load(data_folder+'traces_0.npy', mmap_mode="r")[:,2990+c:3830+c]

	# USE FOR ANTENNA TRACES
	#traces = min_max_norm(traces)

	# standardization of traces, column-wise, for all trace points - USE FOR CABLE TRACES
	mean = np.mean(traces, axis=0)
	std = np.std(traces, axis=0)
	traces_std = (traces-mean)/std

	for i in range(1,N):
	
		traces_tmp = np.load(data_folder+'traces_'+str(i)+'.npy', mmap_mode="r")[:,2990+c:3830+c]  

		# USE FOR ANTENNA TRACES
		#traces_tmp = min_max_norm(traces_tmp)

		# standardization of traces, column-wise, for all trace points - USE FOR CABLE TRACES
		mean = np.mean(traces_tmp, axis=0)
		std = np.std(traces_tmp, axis=0)
		traces_tmp = (traces_tmp-mean)/std
 
		traces = np.append(traces,traces_tmp,axis=0)

	print("shape of traces",traces.shape)
	
	np.save(data_folder_cw+"2024_traces_std.npy", traces) # comment out standardization/min_max scaling above to save unscaled traces
	print("saved traces")
	 
	return 

# Start of execution
init = time.time()

load_traces()

end = time.time()

print("The total running time was: ",(end-init), " seconds.") 
