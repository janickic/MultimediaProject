#cmpt365 project
import math
import cv2
import numpy as np


from Tkinter import *
from tkMessageBox import *
from tkFileDialog   import askopenfilename 
from PIL import Image, ImageTk
import matplotlib.pyplot as plt




# global vars

root = Tk()
filename = ['']
v = IntVar()
v.set(0) #initially set to rows
l_t = Label(root, text=None)
l = Label(root, image=None)
l2 = Label(root, image=None)

def confGui():
	root.title("CMPT 365 Project")
	root.geometry('600x600')
	root.configure(background='grey')


def chooseVideo():
	name= askopenfilename(filetypes=(("All files", "*.*"),("Video files", "*.mp4;*.mpg;*.avi")))
	print(name)
	filename[0]=name

def copyCol():
	l_t.configure(text="Computing...")
	l_t.pack()	
	root.update()

	cap = cv2.VideoCapture(filename[0])
	if not cap.isOpened():
		print("can't open the file")
		return 

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
	num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# print width #352
	# print height #288

	blank_image = np.zeros((height,num_frames,3), np.uint8)

	count_frame=0
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			col =  frame[:,width/2]
			blank_image[:,count_frame] = col
			# print len(frame) # 288
			# print len(frame[0]) #352
			# Our operations on the frame come here
			color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		    # Display the resulting frame
			cv2.imshow('Collecting columns',color)
			count_frame = count_frame + 1

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
		else:
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

	img = Image.fromarray(blank_image, 'RGB')
	img.save('ColCpyImg.png')
	# img.show()
	photo = cv2.imread('ColCpyImg.png',cv2.IMREAD_COLOR)
	cv2.imshow('Hit space to exit',photo)
	cv2.waitKey(0) & 0xFF
	# cv2.waitKey(0)
	cv2.destroyAllWindows()

	l_t.configure(text="Please choose again")
	l_t.pack()
	root.update()


def copyRow():
	l_t.configure(text="Computing...")
	l_t.pack()	
	root.update()

	cap = cv2.VideoCapture(filename[0])
	if not cap.isOpened():
		print("can't open the file")
		return 

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
	num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	blank_image = np.zeros((num_frames, width,3), np.uint8)
	count_frame=0
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			row =  frame[height/2,:]
			blank_image[count_frame,:] = row
			color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		    # Display the resulting frame
			cv2.imshow('Collecting rows',color)
			count_frame = count_frame + 1
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
		else:
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

	rot = np.rot90(blank_image)
	img = Image.fromarray(rot, 'RGB')
	img.save('RowCpyImg.png')
	# img.show()
	photo = cv2.imread('RowCpyImg.png',cv2.IMREAD_COLOR)
	cv2.imshow('Hit space to exit',photo)
	cv2.waitKey(0) & 0xFF
	# cv2.waitKey(0)
	cv2.destroyAllWindows()

	l_t.configure(text="Please choose again")
	l_t.pack()
	root.update()

def chromaticity(arr):
	a = arr 
	for j in range(len(a)):
	    for i in range(len(a[0])):
	        r, g, b = a[j][i][0], a[j][i][1], a[j][i][2]
	        if(r==0)&(g==0)&(b==0):
	            a[j][i][0], a[j][i][1], a[j][i][2] = 0, 0, 0
	        else:
	            a[j][i][0] = int(round((((a[j][i][0]+0.0)/((r+0)+g+b))*255)))
	            a[j][i][1] = int(round((((a[j][i][1]+0.0)/((r+0)+g+b))*255)))
	            a[j][i][2] = int(round((((a[j][i][2]+0.0)/((r+0)+g+b))*255)))
	return a



def makeHist(img_arr):  
	new_arr = [[],[]]
	for i in range(len(img_arr)):
		new_arr[0].append(img_arr[i][0]) #r
		new_arr[1].append(img_arr[i][1]) #g

	new_arr = np.asarray(new_arr, dtype=np.uint8)   
	bins = int(1 + math.log(len(img_arr), 2)) 
	bin_range = [0, 256, 0, 256] 

	hist = cv2.calcHist(new_arr, [0, 1], None, [bins, bins], bin_range) 
	return hist

# Returns list of histograms
def histList(img_arr):
	histogram_list = []
	run = len(img_arr)
	for i in range(len(img_arr)):
		histogram_list.append(makeHist(img_arr[i]))
    
	return histogram_list


def histCol():
	l_t.configure(text="Please wait, computing may take a while")
	l_t.pack()	
	root.update()
	count = 0   
	cap = cv2.VideoCapture(filename[0]) 

	if not cap.isOpened():
		print("can't open the file")

	while(True):
		try:
			ret, frame = cap.read()
			if not ret:
				break
		except:
			print("Frame failed to load")
        
        # Resize image for faster computation
		frame = cv2.resize(frame, (32, 32))

        # Array A stores chromaticity values
		if count == 0:
			A = [chromaticity([frame[i]]) for i in range(32)]
			count = count + 1
		else:
			for i in range(32):
				A[i].append(frame[i])
    
	for i in range(len(A)):
		A[i] = chromaticity(A[i])
        
    # Column dominant matrix
	column_dominant = np.transpose(A, (2, 1, 0, 3)) 
   
	histogram_list = []
	for i in range(len(A)): 
		chrom_list = np.asarray(column_dominant[0], dtype=np.uint8) 
		L = histList(chrom_list)
		# Find difference in histograms
		L2 = []
		for j in range((len(L))-1):
			L2.append(cv2.compareHist(L[j],L[j+1],2))

		histogram_list.append(L2)
	histogram_list = np.dot(histogram_list, (1.0/32))
	histogram_list = np.dot(histogram_list, 255)
	img = Image.fromarray(np.asarray(histogram_list, dtype=np.uint8), "L")

	img.save('HistColImg.png')
	# img.show()
	photo = cv2.imread('HistColImg.png',cv2.IMREAD_COLOR)
	
	cv2.namedWindow('Hit space to exit',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Hit space to exit', 300,300)

	cv2.imshow('Hit space to exit',photo)
	cv2.waitKey(0) & 0xFF
	cv2.destroyAllWindows()

	l_t.configure(text="Please choose again")
	l_t.pack()
	root.update()


def histRow():
	l_t.configure(text="Please wait, computing may take a while")
	l_t.pack()	
	root.update()
	count = 0   
	cap = cv2.VideoCapture(filename[0]) 

	if not cap.isOpened():
		print("can't open the file")

	while(True):
		try:
			ret, frame = cap.read()
			if not ret:
				break
		except:
			print("Frame failed to load")
        
        # Resize image for faster computation
		frame = cv2.resize(frame, (32, 32))
        # Array A stores chromaticity values
		if count == 0:
			A = [chromaticity([frame[i]]) for i in range(32)]
			count = count + 1
		else:
			for i in range(32):
				A[i].append(frame[i])
    
	for i in range(len(A)):
		A[i] = chromaticity(A[i])
        

	histogram_list = []
	for i in range(len(A)): 
		chrom_list = np.asarray(A[i], dtype=np.uint8) 
		L = histList(chrom_list)
		# Find difference in histograms
		L2 = []
		for j in range((len(L))-1):
			L2.append(cv2.compareHist(L[j],L[j+1],2))

		histogram_list.append(L2)
	histogram_list = np.dot(histogram_list, (1.0/32))
	histogram_list = np.dot(histogram_list, 255)
	img = Image.fromarray(np.asarray(histogram_list, dtype=np.uint8), "L")

	img.save('HistRowImg.png')
	# img.show()
	photo = cv2.imread('HistRowImg.png',cv2.IMREAD_COLOR)
	
	cv2.namedWindow('Hit space to exit',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Hit space to exit', 300,300)

	cv2.imshow('Hit space to exit',photo)
	cv2.waitKey(0) & 0xFF
	cv2.destroyAllWindows()

	l_t.configure(text="Please choose again")
	l_t.pack()
	root.update()


def parseVideo(choice):
	if (filename == []) or (filename[0] == ''):
		showinfo("Warning", "You must select a video using the 'Choose Video' button before choosing a function")
		
	else:
		if choice == 1: 
			copyCol()
		elif choice == 2: 
			copyRow()
		elif choice == 3:
			histCol()
		elif choice == 4:
			histRow()

# Open GUI
def main():

	confGui()

	compute_choice = [("Rows", 0), ("Columns", 1)]
	
	Button(text="Choose Video", width = 70, command=lambda : chooseVideo()).pack()
	Button(text="STI by Copying Pixels : Columns", width = 70, command=lambda : parseVideo(1)).pack()
	Button(text="STI by Copying Pixels: Rows", width = 70, command=lambda : parseVideo(2)).pack()
	Button(text="STI by Histogram Differences: Columns", width = 70, command=lambda : parseVideo(3)).pack()
	Button(text="STI by Histogram Differences: Rows", width = 70, command=lambda : parseVideo(4)).pack()
 
	root.mainloop() 

main()