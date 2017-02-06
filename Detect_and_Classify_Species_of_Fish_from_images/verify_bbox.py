import json
import cv2
import numpy as np
import os
import pandas as pd
import csv

from itertools import izip

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def prune_keypoints(kp,des):
	kp_sizes = [];
	for k in kp:
		kp_sizes.append(k.size)
	kp_sizes = np.array(kp_sizes);
	to_keep = kp_sizes>3.0;
	kp_pruned = list(np.array(kp)[to_keep])
	des_pruned = des[to_keep,:]
	return kp_pruned,des_pruned

def prune_keypoints_cluster(kp,des):
	n_clusters_list = [2,3,4];
	scaler = StandardScaler();  
	X = scaler.fit_transform(des);
	kms = []
	inertia = []
	for nc in n_clusters_list:
		km = KMeans(n_clusters=nc,max_iter=1000,tol=1e-8,init='random',n_init=20);
		km.fit(X)
		kms.append(km)
		inertia.append(km.inertia_);
	dinertia = np.diff(inertia)
	km = kms[np.argmin(dinertia)]
	n_clusters = n_clusters_list[np.argmin(dinertia)]
	y = km.labels_;
	cluster_size = np.zeros(n_clusters)
	cluster_uncompacity = np.zeros(n_clusters)
	for i in range(n_clusters):
		nci = (y==i).sum()
		cluster_uncompacity[i] = np.mean(np.linalg.norm(X[y==i,:]-km.cluster_centers_[i,:]));
		cluster_size[i] = nci
	# im = np.argmax(cluster_uncompacity)
	im = np.argmin(cluster_size);
	to_keep = y==im
	kp_pruned = np.array(kp)[to_keep]; 
	des_pruned = des[to_keep]
	return kp_pruned,des_pruned
	

	
labels = ['alb','bet','dol','lag','other','shark','yft']

nof_features = dict();
nof_features_pruned = dict();
features_file = 'sift_features.csv';
try:
	os.remove(features_file);
except:
	pass

header = ['index','image','octave','x','y','angle','response','size']+['feat'+str(i) for i in range(128)]+['label']
formats = [np.int]+['S256']+(len(header)-3)*[np.float]+['S256']
dtype = np.dtype(zip(header,formats));
chunk_size = 100;
chunk = np.zeros((chunk_size,),dtype=dtype);

count_samples = 0;
with open(features_file,'a') as csvfile:
	writer = csv.writer(csvfile);
	writer.writerow(header);
	for l in labels:
	    print l.upper()+'...'
	    files = json.load(open(l+'_labels.json'));
	    try:
		os.mkdir('train/'+l.upper()+'/crop/')
	    except:
		pass
	    os.system('rm -rf train/'+l.upper()+'/crop/*');
	    os.mkdir('train/'+l.upper()+'/crop/pruned')
	    for f in files:
		filename = f['filename'];
		annotations = f['annotations']
		classe = f['class']
		image = cv2.imread('train/'+l.upper()+'/'+filename,-1);
		image_drawing = image.copy()
		image_mask = np.zeros_like(image);
		anno_counter = 0;
		for a in annotations:
			x = int(a['x'])
			x = x-x*(x<0)
			y = int(a['y'])
			y = y-y*(y<0)
			h = int(a['height'])
			w = int(a['width'])
			c = a['class']
			crop = image[y:y+h,x:x+w,:];
			name = 'train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(anno_counter)+'.jpg'
			success = cv2.imwrite(name,crop);
			cv2.rectangle(image_drawing,(x,y),(x+w,y+h),(0,0,255),3)
			cv2.rectangle(image_mask,(x,y),(x+w,y+h),(255,255,255),thickness=cv2.cv.CV_FILLED)
			success = cv2.imwrite('train/'+l.upper()+'/'+filename[:-4]+'_gt.jpg',image_drawing);
			success = cv2.imwrite('train/'+l.upper()+'/'+filename[:-4]+'_m.jpg',image_mask.astype(np.uint8));
			# compute SIFT keypoint and feature
			## get gray image
			gray = cv2.cvtColor(cv2.imread('train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(anno_counter)+'.jpg',-1),cv2.COLOR_BGR2GRAY);
			## instantiate and call SIFT extractor
			# sift = cv2.SIFT(nOctaveLayers=4,contrastThreshold=0.02,edgeThreshold=10,sigma=2.1);
			sift = cv2.SIFT(nOctaveLayers=4,contrastThreshold=0.02,edgeThreshold=20,sigma=1.2)		
			kp,des = sift.detectAndCompute(gray,None);
			## draw SIFT keypoints
			gray_kp = cv2.drawKeypoints(gray,kp,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
			success = cv2.imwrite('train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(anno_counter)+'_siftkp.jpg',gray_kp);
			## prune and redraw SIFT keypoints
			kp_pruned = kp;
			des_pruned = des;
			#kp_pruned,des_pruned = prune_keypoints(kp,des);
			#if len(kp)>2000:
			#	kp_pruned, des_pruned = prune_keypoints_cluster(kp,des);
			#else:
			#	kp_pruned = kp; des_pruned = des;
			gray_kp_pruned = cv2.drawKeypoints(gray,kp_pruned,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
			success = cv2.imwrite('train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(anno_counter)+'_siftkp_pruned.jpg',gray_kp_pruned);
			#if len(kp)>2000:
			#	success = cv2.imwrite('train/'+l.upper()+'/crop/pruned/'+filename[:-4]+'_'+str(anno_counter)+'_siftkp.jpg',gray_kp);
			#	success = cv2.imwrite('train/'+l.upper()+'/crop/pruned/'+filename[:-4]+'_'+str(anno_counter)+'_siftkp_pruned.jpg',gray_kp_pruned);
			## count number of features per image
			nof_features['train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(anno_counter)+'_siftkp.jpg']=len(kp);
			nof_features_pruned['train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(anno_counter)+'_siftkp_pruned.jpg']=len(kp_pruned);
			anno_counter = anno_counter+1
			for ki,di in izip(kp_pruned,des_pruned):
				count_samples = count_samples + 1;
				fi = tuple([count_samples-1,name,ki.octave,ki.pt[0],ki.pt[1],ki.response,ki.size,ki.angle]+di.tolist()+[l.upper()]);
				chunk[(count_samples-1)%chunk_size] = fi;
				if (count_samples-1)%chunk_size == (chunk_size-1):
					pd.DataFrame(chunk).to_csv(csvfile,header=False,index=False);

csvfile.close();


