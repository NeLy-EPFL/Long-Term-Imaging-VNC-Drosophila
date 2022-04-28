import numpy as np
import cv2
from skimage import io
import heapq
import os
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats


def extract_contours(image_curr,top_contour,bottom_contour,blur_val):
	"""
	This function thresholds the template image and detects all the contours on this thresholded image. 
	It returns the contours associated with two regions of interest (ROIs) in our 2-photon images.

	Parameters
	----------
	image_curr: the template image to extract the contour from
	top_contour: the index of the first contour to extract
	bottom_contour: the index of the second contour to extract
	blur_val: the value used to blur the thresholded image before detecting the contours with Open CV

	Returns
	-------
	top_contour: the contour of the first ROI
	bottom_contour: the contour of the second ROI

	"""
	image_curr_conv = cv2.cvtColor(image_curr, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(image_curr_conv, 60, 255, cv2.THRESH_BINARY)
	thresh = cv2.blur(thresh,(blur_val,blur_val))
	contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
	sizes=[]
	for m in range(len(contours)):
		sizes.append(contours[m].size)
	biggest_contours = heapq.nlargest(10, sizes)
	top_contour = contours[sizes.index(biggest_contours[top_contour])]
	bottom_contour = contours[sizes.index(biggest_contours[bottom_contour])]
	return top_contour, bottom_contour

def extract_mask(image_template,top_ctr, btm_ctr):
	"""
	This function creates a mask image that will be used later to extract the mean fluorescence values in the selected ROIs. 

	Parameters
	----------
	image_template: the template image that was used to extract the contours
	top_ctr: the contour of the first ROI
	btm_ctr: the contour of the second ROI

	Returns
	-------
	mask_top: a mask image containing only the first ROI's contour drawn
	mask_bot: a mask image containing only the second ROI's contour drawn

	"""
	image_template_conv = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
	image_top = image_template_conv.copy()
	image_bot = image_template_conv.copy()
	mask_top = np.zeros(image_top.shape, np.uint8)
	mask_bot = np.zeros(image_bot.shape, np.uint8)
	cv2.drawContours(mask_top, top_ctr, -1, 255, -1)
	cv2.drawContours(mask_bot, btm_ctr, -1, 255, -1)
	return mask_top, mask_bot

def get_mean_values(stack_curr,mask_temp_top, mask_temp_bot):
	"""
	This function calculates the mean pixel intensities from our 2-photon images inside the two previously selected ROIs. 

	Parameters
	----------
	stack_curr: the stack of 2-photon images to calculate the mean pixel intensity from.
	mask_temp_top: the mask image with the first ROI drawn
	mask_temp_bot: the mask image with the second ROI drawn

	Returns
	-------
	values_top: a list containing the mean ROI (first ROI) pixel intensitiy values for each 2-photon image.
	values_bot: a list containing the mean ROI (second ROI) pixel intensitiy values for each 2-photon image.

	"""

	values_top = []
	values_bot = []
	for m in range(len(stack_curr)):
		mean_top = cv2.mean(stack_curr[m], mask=mask_temp_top)
		values_top.append(mean_top[1])
		mean_bot = cv2.mean(stack_curr[m], mask=mask_temp_bot)
		values_bot.append(mean_bot[1])

	return values_top, values_bot

def save_images(stack_to_save,top_contour,bottom_contour,path_folder):
	"""
	This function draws the 2 selected ROIs on each 2-photon image and saves it in a newly created directory.

	Parameters
	----------
	stack_to_save: the stack of 2-photon images to calculate the mean pixel intensity from.
	top_contour: the first ROI.
	bottom_contour: the second ROI.
	path_folder: the folder path in which the newly directory will be created

	Returns
	-------

	"""
	save_dir = (path_folder+ "/roi_stack/")
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	for m in range(len(stack_to_save)):
		cv2.drawContours(image=stack_to_save[m], contours=top_contour, contourIdx=-1, color=(21,237,237), thickness=2, lineType=cv2.LINE_AA)
		cv2.drawContours(image=stack_to_save[m], contours=bottom_contour, contourIdx=-1, color=(237, 74, 233), thickness=2, lineType=cv2.LINE_AA)
		io.imsave(save_dir+str(m)+".tif",stack_to_save[m])

	return

def get_box_plot_info(days_list,control_top_all,control_bot_all,leg_removed_top_all,leg_removed_bot_all):
	"""
	This function creates a list per ROI and per fly group. These lists store the ROIs' mean intensity values per group of fly per day. 
	They will be used to create the boxplot next to the raw datapoints.

	Parameters
	----------
	days_list: a list containing the number of days of 2-photon recordings.
	control_top_all: a list that contains one list per fly within that group. Each fly list contains the associated ROI mean pixel values per day. 
	This list stores information about the control group and the first ROI.
	control_bot_all: a list that contains one list per fly within that group. Each fly list contains the associated ROI mean pixel values per day.
	This list stores information about the control group and the second ROI.
	leg_removed_top_all: a list that contains one list per fly within that group. Each fly list contains the associated ROI mean pixel values per day.
	This list stores information about the limb amputation  group and the first ROI.
	leg_removed_bot_all: a list that contains one list per fly within that group. Each fly list contains the associated ROI mean pixel values per day.
	This list stores information about the limb amputation group and the second ROI.

	Returns
	-------
	box_cont_top: a list containing one list per day with all the ROIs mean values for all the flies within that group. This list is for the first ROI control group.
	box_cont_bot: a list containing one list per day with all the ROIs mean values for all the flies within that group. This list is for the second ROI control group.
	box_lr_top: a list containing one list per day with all the ROIs mean values for all the flies within that group. This list is for the first ROI limb amputation group.
	box_cont_top: a list containing one list per day with all the ROIs mean values for all the flies within that group. This list is for the second ROI limb amputation group.
	"""

	box_cont_top = []
	box_cont_bot = []
	box_lr_top = []
	box_lr_bot = []
	for day in days_list:
	    i = day-1
	    cont_top_temp = []
	    cont_bot_temp = []
	    lr_top_temp = []
	    lr_bot_temp = []
	    for m in range(len(control_top_all)):
	    	cont_top_temp.append(control_top_all[m][i])
	    	cont_bot_temp.append(control_bot_all[m][i])
	    	lr_top_temp.append(leg_removed_top_all[m][i])
	    	lr_bot_temp.append(leg_removed_bot_all[m][i])

	    box_cont_top.append(cont_top_temp)
	    box_cont_bot.append(cont_bot_temp)
	    box_lr_top.append(lr_top_temp)
	    box_lr_bot.append(lr_bot_temp)

	box_cont_top = [[x for x in y if not np.isnan(x)] for y in box_cont_top]
	box_cont_bot = [[x for x in y if not np.isnan(x)] for y in box_cont_bot]
	box_lr_top = [[x for x in y if not np.isnan(x)] for y in box_lr_top]
	box_lr_bot = [[x for x in y if not np.isnan(x)] for y in box_lr_bot]

	return box_cont_top, box_cont_bot, box_lr_top, box_lr_bot

def save_figure(path_fig,days, boxplot_cont_top, boxplot_cont_bot, boxplot_lr_top, boxplot_lr_bot,control_top_all,control_bot_all,leg_removed_top_all,leg_removed_bot_all,pvalues_top,pvalues_bottom):
	"""
	This function creates the 2 final plots presented in Figure 3. One plot is associated with the first ROI while the second one is associated with the second ROI. 
	Each figure shows the ROIs mean fluorescence values per day per fly. Flies are separated into two colours (control group in orange and limb amputation in blue). 
	"""
	selectedX = [item*4 for item in days]
	shift_selectedX1 = [item-3.4 for item in selectedX]
	shift_selectedX2 = [item-2.5 for item in selectedX]
	shift_selectedX3 = [item-1.6 for item in selectedX]
	shift_selectedX4 = [item-0.9 for item in selectedX]
	labels = [str(item) for item in days]
	labels.insert(0,'0')

	figure = plt.figure(1)
	ax = plt.subplot(1, 1, 1)
	figure.set_size_inches(14,8)
	plt.boxplot(boxplot_cont_top,positions=shift_selectedX2,widths=0.22,medianprops=dict(color="black"),showfliers=False)
	plt.boxplot(boxplot_lr_top,positions=shift_selectedX4,widths=0.22,medianprops=dict(color="black"),showfliers=False)
	plt.plot(shift_selectedX1, control_top_all[0], 'v', color='lightsteelblue',label='C1',markersize=4)
	plt.plot(shift_selectedX1, control_top_all[1], 'v', color='mediumslateblue',label='C2',markersize=4)
	plt.plot(shift_selectedX1, control_top_all[2], 'v', color='darkblue',label='C3',markersize=4)
	plt.plot(shift_selectedX1, control_top_all[3], 'v', color='royalblue',label='C4',markersize=4)
	plt.plot(shift_selectedX1, control_top_all[4], 'v', color='dodgerblue',label='C5',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_top_all[0], 'o', color='darkorange',label='LR1',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_top_all[1], 'o', color='sandybrown',label='LR2',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_top_all[2], 'o', color='moccasin',label='LR3',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_top_all[3], 'o', color='tomato',label='LR4',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_top_all[4], 'o', color='orangered',label='LR5',markersize=4)
	for idx, pv in enumerate(pvalues_top):
		plt.text(x=idx*4, y=134, s=pv)
	plt.xticks(days) 
	plt.ylim(0,140)
	ax.xaxis.set_major_locator(ticker.MaxNLocator(18))
	ax.set_xticklabels(labels)
	plt.ylabel("$\%$ of fluorescence")
	plt.xlabel("Days")
	plt.legend(loc='lower left')
	plt.savefig(path_fig+'/_contours_top.eps', format='eps')

	figure = plt.figure(2)
	ax = plt.subplot(1, 1, 1)
	figure.set_size_inches(14,8)
	plt.plot(shift_selectedX1, control_bot_all[0], 'v', color='lightsteelblue',label='C1',markersize=4)
	plt.plot(shift_selectedX1, control_bot_all[1], 'v', color='mediumslateblue',label='C2',markersize=4)
	plt.plot(shift_selectedX1, control_bot_all[2], 'v', color='darkblue',label='C3',markersize=4)
	plt.plot(shift_selectedX1, control_bot_all[3], 'v', color='royalblue',label='C4',markersize=4)
	plt.plot(shift_selectedX1, control_bot_all[4], 'v', color='dodgerblue',label='C5',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_bot_all[0], 'o', color='darkorange',label='LR1',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_bot_all[1], 'o', color='sandybrown',label='LR2',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_bot_all[2], 'o', color='moccasin',label='LR3',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_bot_all[3], 'o', color='tomato',label='LR4',markersize=4)
	plt.plot(shift_selectedX3, leg_removed_bot_all[4], 'o', color='orangered',label='LR5',markersize=4)
	plt.boxplot(boxplot_cont_bot,positions=shift_selectedX2,widths=0.22,medianprops=dict(color="black"),showfliers=False)
	plt.boxplot(boxplot_lr_bot,positions=shift_selectedX4,widths=0.22,medianprops=dict(color="black"),showfliers=False)
	for idx, pv in enumerate(pvalues_bottom):
		plt.text(x=idx*4, y=134, s=pv)
	plt.ylim(0,140)
	ax.xaxis.set_major_locator(ticker.MaxNLocator(18))
	ax.set_xticklabels(labels)
	plt.ylabel("$\%$ of fluorescence")
	plt.xlabel("Days")
	plt.legend(loc='lower left')
	plt.savefig(path_fig+'/_contours_bottom.eps', format='eps')

	return 

def get_p_value(control_values_per_day,legremoved_values_per_day):
	pval_day_summary = [" "]
	for i in range(1,len(control_values_per_day)):
		statval, pval = scipy.stats.mannwhitneyu(control_values_per_day[i],legremoved_values_per_day[i])
		if pval < 0.01:
			pval_day_summary.append("**")
		elif pval < 0.05:
			pval_day_summary.append("*")
		else:
			pval_day_summary.append(" ")
	return pval_day_summary

#### MAIN ####
#initiating the contours index values per fly and the blurring values used to extract the contours in our 2-photon images.
LR_blur = [3,3,4,2,2]
LR_top = [1,0,0,2,1]
LR_bottom = [3,3,3,4,4]
control_blur = [3,3,3,2,2]
control_top = [0,1,1,0,1]
control_bottom = [3,2,3,3,3]
control_top_all = [[],[],[],[],[]]
control_bot_all = [[],[],[],[],[]]
leg_removed_top_all = [[],[],[],[],[]]
leg_removed_bot_all = [[],[],[],[],[]]

#Change this path to the folder's path containing all the control and limb amputation data. Do not change the flies numbering after downloading the data.
rootdir = "" #"PATH TO FLIES FOLDERS"

#Looping over the flies' directories 
for dirs in next(os.walk(rootdir))[1]:
	curr_folder = os.path.join(rootdir, dirs)
	print("curr_folder",curr_folder)
	#if conditions to select the appropriate contours indexes and blurring values.
	if "Control" in dirs:
		bool_control = True
		fly_val = int(dirs.split('_fly')[1])
		curr_blur = control_blur[fly_val-1]
		curr_top = control_top[fly_val-1]
		curr_bottom = control_bottom[fly_val-1]
	if "LR" in dirs:
		bool_control = False
		fly_val = int(dirs.split('_fly')[1])
		curr_blur = LR_blur[fly_val-1]
		curr_top = LR_top[fly_val-1]
		curr_bottom = LR_bottom[fly_val-1]
	#Limb amputation fly 3 needed two template images to detect the appropriate contours, the steps are the same as explained on line #279
	if bool_control==False and fly_val==3:
		imagetop = io.imread(curr_folder + "/Template.tif")
		imagebot = io.imread(curr_folder + "/TemplateBot.tif")
		stack_curr = io.imread(curr_folder + "/Aligned.tif")
		top_contours, wrong = extract_contours(imagetop,curr_top,curr_bottom,curr_blur,)
		wrong, bottom_contours = extract_contours(imagebot,curr_top,2,curr_blur,)
		top_mask_template, bottom_mask_template = extract_mask(imagetop,top_contours,bottom_contours)
		curr_mean_top, curr_mean_bot = get_mean_values(stack_curr,top_mask_template,bottom_mask_template)
		save_images(stack_curr,top_contours,bottom_contours,curr_folder)
	#Control fly 3 needed two different blurring values to detect the appropriate contours, the steps are the same as explained on line #279	
	elif bool_control==True and fly_val==3:
		image = io.imread(curr_folder + "/Template.tif")
		stack_curr = io.imread(curr_folder + "/Aligned.tif")
		top_contours, wrong = extract_contours(image,curr_top,curr_bottom,curr_blur,)
		wrong, bottom_contours = extract_contours(image,curr_top,curr_bottom,2,)
		top_mask_template, bottom_mask_template = extract_mask(image,top_contours,bottom_contours)
		curr_mean_top, curr_mean_bot = get_mean_values(stack_curr,top_mask_template,bottom_mask_template)
		save_images(stack_curr,top_contours,bottom_contours,curr_folder)	
	#For each fly, the template image is loaded and the stack of 2-photon images is also loaded. The contours are detected and mask images are created.
	#Mean pixel values within the ROIs are extracted from each 2-photon image. The contours are then drawn on each 2-photon images. 
	else:
		image = io.imread(curr_folder + "/Template.tif")
		stack_curr = io.imread(curr_folder + "/Aligned.tif")
		top_contours, bottom_contours = extract_contours(image,curr_top,curr_bottom,curr_blur,)
		top_mask_template, bottom_mask_template = extract_mask(image,top_contours,bottom_contours)
		curr_mean_top, curr_mean_bot = get_mean_values(stack_curr,top_mask_template,bottom_mask_template)
		save_images(stack_curr,top_contours,bottom_contours,curr_folder)
#"""
#"""
	#inserting nan values for some missing recordings.
	if bool_control == True:
		if fly_val == 1:
			curr_mean_top.insert(2,np.nan)
			curr_mean_bot.insert(2,np.nan)
		if fly_val == 3:
			curr_mean_top.insert(8,np.nan)
			curr_mean_bot.insert(8,np.nan)
		if fly_val == 4:
			curr_mean_top.insert(7,np.nan)
			curr_mean_bot.insert(7,np.nan)
		#each extracted fluorescence value is divided to the first day's fluorescence value and multiplied by 100.
		curr_mean_top = [(item/curr_mean_top[0])*100 for item in curr_mean_top]
		curr_mean_bot = [(item/curr_mean_bot[0])*100 for item in curr_mean_bot]
		control_top_all[fly_val-1] = curr_mean_top
		control_bot_all[fly_val-1] = curr_mean_bot

	if bool_control == False:
		if fly_val == 2:
			curr_mean_top.insert(5,np.nan)
			curr_mean_bot.insert(5,np.nan)
		if fly_val == 5:
			curr_mean_top.insert(1,np.nan)
			curr_mean_bot.insert(1,np.nan)
			curr_mean_top.insert(2,np.nan)
			curr_mean_bot.insert(2,np.nan)
			curr_mean_top.insert(11,np.nan)
			curr_mean_bot.insert(11,np.nan)	
		#each extracted fluorescence value is divided to the first day's fluorescence value and multiplied by 100.
		curr_mean_top = [(item/curr_mean_top[0])*100 for item in curr_mean_top]
		curr_mean_bot = [(item/curr_mean_bot[0])*100 for item in curr_mean_bot]
		leg_removed_top_all[fly_val-1] = curr_mean_top
		leg_removed_bot_all[fly_val-1] = curr_mean_bot

#extracting the ROIs mean fluorescence values per day and plotting the final figures showing the ROIs mean fluorescence values per control group and their associated boxplots.

days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
boxplot_cont_top, boxplot_cont_bot, boxplot_lr_top, boxplot_lr_bot = get_box_plot_info(days,control_top_all,control_bot_all,leg_removed_top_all,leg_removed_bot_all)
pvalues_top = get_p_value(boxplot_cont_top,boxplot_lr_top)
pvalues_bottom = get_p_value(boxplot_cont_bot,boxplot_lr_bot)
save_figure(rootdir, days, boxplot_cont_top, boxplot_cont_bot, boxplot_lr_top, boxplot_lr_bot,control_top_all,control_bot_all,leg_removed_top_all,leg_removed_bot_all,pvalues_top,pvalues_bottom)


