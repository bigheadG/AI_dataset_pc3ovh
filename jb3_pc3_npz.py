#=============================================
# File Name: jb2_pc3_npz.py
#
#   modify from: jb2_pc3_parsing_wsliding.py
#
# this file is for generate a series of sliding window based on width
# and step 
#
# Requirement:
# Hardware: BM501-AOP
# Firmware: pc3
# Out put X,Y
#
# modify setting from: mmWave_uDoppler.json
# input file: process all of uDopoller___.csv in the ./2_dataset_s  directory
# output: dataset.npz
# output image files in ./2_image directory
#
#=============================================

import numpy as np
import pandas as pd
import csv
import pandas as pd
from tensorflow.keras.utils import to_categorical
import os
import json
import time

import imageio as im

import JBUIWidget as jbui

########### read json file ######
with open('jb0_pc3_control.json') as f:
	jdata = json.load(f)

################### AI parsing data parameter setting   ######
PROGRAM_FUNC = jdata["mmWave"]["working_mode"]["functions"]
WORKING_MODE = jdata["mmWave"]["working_mode"]["select"]  #select: real-time: 1  playback: 0   recording: 2  runtime(AI):3  AI_labeling:4
FUNCTION = PROGRAM_FUNC[WORKING_MODE]
JB_logFrames = jdata["mmWave"]["AI_parsing_setting"]["logFrames"]      #500 for 20 sec log data
JB_logWidth  = jdata["mmWave"]["AI_parsing_setting"]["logWidth"]       #1000 log data width
JB_offset    = jdata["mmWave"]["AI_parsing_setting"]["offset"]         #window starting address 
JB_width     = jdata["mmWave"]["AI_parsing_setting"]["window_width"]   #50 for 2 sec, frames per sample
JB_step      = jdata["mmWave"]["AI_parsing_setting"]["step"]           #10 for wondow slide
JB_datasetArrayName = jdata["mmWave"]["AI_parsing_setting"]["datasetArrayName"]  # output default file name: 'dataset.npz'
JB_labelingType = jdata["mmWave"]["working_mode"]["labelingType"]
JB_version   = jdata["mmWave"]["info"]["version"] 
JB_col_name = ['frameNum','labelType','action_id','doppler']
jb_path = os.getcwd() + '/1_dataset/'

path = './1_dataset/'
imgPath = './1_dataset/2_image/'
wspath = './1_dataset/2_dataset_s/'

print("\n************** {:} *************\n".format(JB_version))
print("Output file: {:}".format(jb_path + JB_datasetArrayName))
print("Output image file: {:}*.jpg".format(imgPath))
print("*******************************************************************")
print("JB> Starting process time: {:}".format(time.ctime()))  



def jb_fileMerged():
	global wspath,path
	path = wspath
	jb_fileNum = 0
	jb_output_file = path + 'jb_merge.csv'
	df = pd.DataFrame([], columns=JB_col_name) # init
	i = 0
	for jb_file in os.listdir(path):
		if jb_file.startswith('uDoppler_') and jb_file.endswith('.csv'):
			print('JB> input file name={}'.format(jb_file))
			col_names = JB_col_name #['frameName','labelType','action_id','doppler']
			df_tmp = pd.read_csv(path + "/"+jb_file, names = col_names, skiprows = [0],
				  dtype={'frameNum': int,'labelType' : int, 'action_id':int}) 
			df_tmp.dropna()			
			df = df.append(df_tmp, ignore_index = True)
			i += 1
	df.to_csv(jb_output_file)
	print('--------------------------------------')
	print('JB> OK! output file name={} :Totals:{:}'.format(jb_output_file,i))
	print('--------------------------------------')
	return df

tt = time.localtime()
dt = time.strftime("%Y-%m-%d-%H-%M-%S",tt) 

JB_num = 0
 

def findStringInFile(str = None):
    fns = []
    for fname in os.listdir('.'):    # change directory as needed
        if os.path.isfile(fname):    # make sure it's a file, not a directory entry
            with open(fname) as f:   # open file
                i = 0 
                for line in f:       # process line by line
                    i+=1
                    if i == 2:
                        s = line.split(",")
                        if s[0] == str :
                            fns.append(fname)
                        break
    return fns
    

def dataLabeling():
	df = jb_fileMerged()
	fnSet = set(df.frameNum)
	print(fnSet)
	lsA = []
	pictureA = []
	for i in fnSet:
		data = df[(df.frameNum == i)]
		labelS = set(data.labelType)
		label = labelS.pop()
		windowA = data.loc[:,['doppler']]
		print("label:{:}".format(label))
		wsA =np.empty((0,))
		for iw in range(len(windowA)): # windowA contains of the width * 500(points)
			dA = np.array([float(j) for j in windowA.iat[iw,0].replace('[', '').replace(']', '').replace("'",'').split(",")])
			#print("dA.shape={:}  wsA:{:}".format(dA.shape,wsA.shape))
			wsA = np.concatenate((wsA, dA))
						
			print("JB> a action:{:}  wsA: {:}   JB_logWidth:{:}".format(JB_labelingType[label],wsA.shape,JB_logWidth))
			
			#wSlideA.append(wsAr)
		lsA.append(label)
		wsAr = wsA.reshape(JB_width, JB_logWidth)
		
		pictureA.append(wsAr)
		print("lsA:{:} lsA:{:} wsAr.shape:{:}".format(np.array(lsA).shape,lsA,wsAr.shape))
	
	
	
	lsA_np = np.array(lsA)
	pictureA_np = np.array(pictureA)
	
	print("lsA_np={:}  pictureS_np:{:}".format(lsA_np.shape,pictureA_np.shape))
	X = pictureA_np
	Y = to_categorical(lsA_np) # 
	
	return X,Y

X, Y = dataLabeling()





def extractSlideWindow(fileName, width = None, step = None):
	global JB_logFrames, JB_width,JB_logWidth

	df = jb_fileMerged() # overwrite filename here
	#col_names = ['frameNum','labelType','action_id','doppler']
	#df = pd.read_csv(fileName, names = col_names, skiprows = [0])
	#print(df.info()) 
	#print(df.head())
	labelsSet = set(df.labelType)
	labelsSet = {0, 1, 2, 3, 4, 5} # tmp
	#print("\n\nJB_extract> ****************************")
	print("JB_extract> labels: {:}".format(labelsSet))
	#print("  JB_extract> df.labelType: {:}".format(df.labelType))
	lsA = []
	actionMA = []
	for label in labelsSet: 
		actionA = df[(df.labelType == label)]		
		actionsSet =  set(actionA.action_id)
		#print('\nJB_extract> actionsSet={}'.format(actionsSet))		
		#print("  JB_extract> label: {:}    actions: {:}".format(label, actionsSet))		
		#sliding window Array
		for action in actionsSet:
			#print(JB_labelingType)
			#print("  JB_extract> label={:}, action={:5d}".format(JB_labelingType[label], action))
			iA = actionA[(actionA.action_id == action)]
			fA= iA.loc[:,['frameNum']]
			print("JB> frameNumber set:{:}".format(set(iA.frameNum)))
			frameNumber = fA.values[0][0] #fA.loc[0].iat[0]
			print("frameNumber={:}".format(frameNumber))
			dopplerA = iA.loc[:,['doppler']]
			dataLen = len(dopplerA)
			#print("JB dopplerA.len = {:}".format(dataLen))
			#print("    JB_extract> ===dopplerA shape ==:{:}".format(dopplerA.shape))
			#print("    JB_extract> label: {:}   action: {:}  shape:{:} doppler:{:}".format(label,action,iA.shape, dopplerA.shape))
			#sliding step array ex.(step= 10) [0,10,20,30,......] 
			#stepA = [i for i in np.arange(JB_offset, JB_logFrames, step) if i + width <  JB_logFrames + JB_width] # 550
			
			# test, added =
			#stepA = [i for i in np.arange(JB_offset, JB_logFrames, step) if i + width <=  dataLen] # 550
			stepA = [0] 
			print("JB_Step(raw): {:} ".format(stepA))
			wSlideA = [] # np.empty((0,))
			if len(stepA) != 0:
				#print("JB_Step: {:} ".format(stepA))
				#500 item for doppler
				#print("  JB_extract> label={:}, action_id={:5d}   pictures:{:}".format(JB_labelingType[label], action,len(stepA)))

				for si in stepA: 
					wsA =np.empty((0,))
					windowA = dopplerA.iloc[si:si+width]
					
					for iw in range(len(windowA)): # windowA contains of the width * 500(points)
						 
						dA = np.array([float(j) for j in windowA.iat[iw,0].replace('[', '').replace(']', '').replace("'",'').split(",")])
						#print("dA.shape={:}  wsA:{:}".format(dA.shape,wsA.shape))
						wsA = np.concatenate((wsA, dA))
						
					print("JB> a stepA = {:} action:{:}  wsA: {:}  width:{:}  JB_logWidth:{:}".format(len(stepA),JB_labelingType[label],wsA.shape,width,JB_logWidth))
					wsAr = wsA.reshape(width, JB_logWidth)
					print("JB> b action:{:}  wsA: {:} a window of Dopplers Data(wsAr):{:}".format(JB_labelingType[label],wsA.shape,wsAr.shape))
					
					wSlideA.append(wsAr)
					lsA.append(label)
					actionMA.append(wsAr)
					#fileName = "{:}_{:}_{:}_a{:}".format(JB_labelingType[label],frameNumber,si,action)
					#saveImage(matrix= wsAr,frN= fileName)
					
					
					
			ws_np = np.array(wSlideA)
			#print("    JB_extract> wSlide_np:{:}".format(ws_np.shape))		
		actionMA_np = np.array(actionMA)
		#print("  JB_extract> actionMA_np:{:}".format(actionMA_np.shape))
        #lsA.append(wsAr)
	lsA_np = np.array(lsA)
	#print('JB_extract> lsA_np={}'.format(lsA_np)) 	
	#print("JB_extract> lsA_np:{:}, len(lsA):{:}".format(lsA_np.shape, len(lsA)))
	#print("JB_extract> (# of pics, picSize(50,500), dopplerWaveform:500 points) ") 
	#print("JB_extract> samples={}".format(actionMA_np.shape))
	X = actionMA_np
	Y = to_categorical(lsA_np)
	#print("  JB_extract> X.shape={}".format(X.shape))
	#print("  JB_extract> Y.shape={}".format(Y.shape))
	return X, Y

#
# main()
#
#X, Y = extractSlideWindow('', width=JB_width,step = JB_step)
 


print("JB> X.shape={}".format(X.shape))
print("JB> Y.shape={}".format(Y.shape))
# save to file
np.savez(jb_path + JB_datasetArrayName, X, Y, dtpye='float')
dataset = np.load(jb_path + JB_datasetArrayName)
print('JB> read back and checking X shape={}'.format(dataset['arr_0'].shape))
print('JB> read back and checking Y shape={}'.format(dataset['arr_1'].shape))
print('JB> output file name={}'.format(jb_path + JB_datasetArrayName))
#print('JB> Output image file: {:}*.jpg'.format(imgPath))
print("JB> End of process time: {:}".format(time.ctime()))  
print("*******************************************************************")

print("\n*******************************************************************")
print("JB> Please run jb_genModel.py for training model") 
print("*******************************************************************")





