from skimage.io import imread
from skimage import color
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted
from sklearn import linear_model, tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class ECG:
	def _ensure_rgb(self, image):
		if image is None:
			raise ValueError("Unable to read the uploaded image.")
		if image.ndim == 2:
			image = np.stack((image,) * 3, axis=-1)
		elif image.ndim == 3 and image.shape[-1] == 4:
			image = image[:, :, :3]
		elif image.ndim != 3 or image.shape[-1] not in (3,):
			raise ValueError(f"Unsupported image shape: {image.shape}")
		return image

	def _standardize_layout(self, image):
		resized = resize(image, (1572, 2213), preserve_range=True, anti_aliasing=True)
		if np.issubdtype(image.dtype, np.integer):
			return np.clip(resized, 0, 255).astype(np.uint8)
		return np.clip(resized, 0.0, 1.0)

	def _validate_leads(self, leads):
		for idx, lead in enumerate(leads, start=1):
			if lead.size == 0:
				raise ValueError(
					f"Could not extract lead {idx}. The ECG layout does not match the expected 12-lead report format."
				)

	def  getImage(self,image):
		image=imread(image)
		image = self._ensure_rgb(image)
		image = self._standardize_layout(image)
		return image

	def GrayImgae(self,image):
		image = self._ensure_rgb(image)
		image_gray = color.rgb2gray(image)
		image_gray=resize(image_gray,(1572,2213))
		return image_gray

	def DividingLeads(self,image):
		image = self._ensure_rgb(image)
		image = self._standardize_layout(image)
		Lead_1 = image[300:600, 150:643] 
		Lead_2 = image[300:600, 646:1135] 
		Lead_3 = image[300:600, 1140:1625] 
		Lead_4 = image[300:600, 1630:2125] 
		Lead_5 = image[600:900, 150:643] 
		Lead_6 = image[600:900, 646:1135] 
		Lead_7 = image[600:900, 1140:1625] 
		Lead_8 = image[600:900, 1630:2125] 
		Lead_9 = image[900:1200, 150:643] 
		Lead_10 = image[900:1200, 646:1135] 
		Lead_11 = image[900:1200, 1140:1625] 
		Lead_12 = image[900:1200, 1630:2125] 
		Lead_13 = image[1250:1480, 150:2125] 

		
		Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]
		self._validate_leads(Leads)
		fig , ax = plt.subplots(4,3)
		fig.set_size_inches(10, 10)
		x_counter=0
		y_counter=0

	

		for x,y in enumerate(Leads[:len(Leads)-1]):
			if (x+1)%3==0:
				ax[x_counter][y_counter].imshow(y)
				ax[x_counter][y_counter].axis('off')
				ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax[x_counter][y_counter].imshow(y)
				ax[x_counter][y_counter].axis('off')
				ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
				y_counter+=1
	    
	
		fig.savefig('Leads_1-12_figure.png')
		fig1 , ax1 = plt.subplots()
		fig1.set_size_inches(10, 10)
		ax1.imshow(Lead_13)
		ax1.set_title("Leads 13")
		ax1.axis('off')
		fig1.savefig('Long_Lead_13_figure.png')

		return Leads

	def PreprocessingLeads(self,Leads):

		fig2 , ax2 = plt.subplots(4,3)
		fig2.set_size_inches(10, 10)
		x_counter=0
		y_counter=0

		for x,y in enumerate(Leads[:len(Leads)-1]):
			if y.size == 0:
				continue
			grayscale = color.rgb2gray(y)
			blurred_image = gaussian(grayscale, sigma=1)
			global_thresh = threshold_otsu(blurred_image)

			binary_global = blurred_image < global_thresh
			binary_global = resize(binary_global, (300, 450))
			if (x+1)%3==0:
				ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
				ax2[x_counter][y_counter].axis('off')
				ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
				ax2[x_counter][y_counter].axis('off')
				ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
				y_counter+=1
		fig2.savefig('Preprossed_Leads_1-12_figure.png')

		fig3 , ax3 = plt.subplots()
		fig3.set_size_inches(10, 10)
		if Leads[-1].size == 0:
			return
		grayscale = color.rgb2gray(Leads[-1])
		blurred_image = gaussian(grayscale, sigma=1)
		global_thresh = threshold_otsu(blurred_image)
		print(global_thresh)
		binary_global = blurred_image < global_thresh
		ax3.imshow(binary_global,cmap='gray')
		ax3.set_title("Leads 13")
		ax3.axis('off')
		fig3.savefig('Preprossed_Leads_13_figure.png')


	def SignalExtraction_Scaling(self,Leads):
		fig4 , ax4 = plt.subplots(4,3)
		x_counter=0
		y_counter=0
		self.clear_lead_csvs()
		for x,y in enumerate(Leads[:len(Leads)-1]):
			if y.size == 0:
				continue
			grayscale = color.rgb2gray(y)
			blurred_image = gaussian(grayscale, sigma=0.7)
			global_thresh = threshold_otsu(blurred_image)

			binary_global = blurred_image < global_thresh
			binary_global = resize(binary_global, (300, 450))
			contours = measure.find_contours(binary_global,0.8)
			if not contours:
				continue
			contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
			test = None
			for contour in contours:
				if contour.shape in contours_shape:
					test = resize(contour, (255, 2))
			if test is None:
				continue
			if (x+1)%3==0:
				ax4[x_counter][y_counter].invert_yaxis()
				ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
				ax4[x_counter][y_counter].axis('image')
				ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax4[x_counter][y_counter].invert_yaxis()
				ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
				ax4[x_counter][y_counter].axis('image')
				ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
				y_counter+=1
	    
			lead_no=x
			scaler = MinMaxScaler()
			fit_transform_data = scaler.fit_transform(test)
			Normalized_Scaled=pd.DataFrame(fit_transform_data[:,0], columns = ['X'])
			Normalized_Scaled=Normalized_Scaled.T
			if (os.path.isfile('scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no+1))):
				Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1), mode='a',index=False)
			else:
				Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)
	      
		fig4.savefig('Contour_Leads_1-12_figure.png')

	def clear_lead_csvs(self):
		for lead_idx in range(1, 13):
			file_name = f"Scaled_1DLead_{lead_idx}.csv"
			if os.path.exists(file_name):
				os.remove(file_name)

	def CombineConvert1Dsignal(self):
		lead_files = [f"Scaled_1DLead_{idx}.csv" for idx in range(1, 13)]
		existing_files = [file_name for file_name in lead_files if os.path.exists(file_name)]
		if not existing_files:
			raise ValueError("Could not extract a usable ECG signal from this image. Please upload a clear 12-lead ECG report.")

		test_final = pd.read_csv(existing_files[0])
		for file_name in existing_files[1:]:
			df = pd.read_csv(file_name)
			test_final = pd.concat([test_final, df], axis=1, ignore_index=True)

		if test_final.empty:
			raise ValueError("ECG signal extraction returned empty data. Please upload a clearer ECG image.")
		return test_final
		
	def DimensionalReduciton(self,test_final):
		pca_loaded_model = joblib.load('PCA_ECG.pkl')
		result = pca_loaded_model.transform(test_final)
		final_df = pd.DataFrame(result)
		return final_df

	def ModelLoad_predict(self,final_df):
		loaded_model = joblib.load('Heart_Disease_Prediction_using_ECG.pkl')
		result = loaded_model.predict(final_df)
		if result[0] == 1:
			return "You ECG corresponds to Myocardial Infarction"
		elif result[0] == 0:
			return "You ECG corresponds to Abnormal Heartbeat"
		elif result[0] == 2:
			return "Your ECG is Normal"
		else:
			return "You ECG corresponds to History of Myocardial Infarction"
