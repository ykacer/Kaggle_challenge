import json
import cv2
import numpy as np
import os


def prune_keypoints(kp,des):
	kp_sizes = [];
	for k in kp:
		kp_sizes.append(k.size)
	kp_sizes = np.array(kp_sizes);
	to_keep = kp_sizes>2.0;
	kp_pruned = list(np.array(kp)[to_keep])
	des_pruned = des[to_keep,:]
	return kp_pruned,des_pruned

	
labels = ['alb','bet','dol','lag','other','shark','yft']

nof_features = dict();
nof_features_pruned = dict();

for l in labels:
    print l.upper()+'...'
    files = json.load(open(l+'_labels.json'));
    try:
        os.mkdir('train/'+l.upper()+'/crop/')
    except:
        pass
    for f in files:
        filename = f['filename'];
        annotations = f['annotations']
        classe = f['class']
        image = cv2.imread('train/'+l.upper()+'/'+filename,-1);
	image_drawing = image.copy()
        image_mask = np.zeros_like(image);
	count = 0;
        for a in annotations:
            x = int(a['x'])
            x = x-x*(x<0)
            y = int(a['y'])
            y = y-y*(y<0)
            h = int(a['height'])
            w = int(a['width'])
            c = a['class']
	    crop = image[y:y+h,x:x+w,:];
	    success = cv2.imwrite('train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(count)+'.jpg',crop);
            cv2.rectangle(image_drawing,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.rectangle(image_mask,(x,y),(x+w,y+h),(255,255,255),thickness=cv2.cv.CV_FILLED)
            success = cv2.imwrite('train/'+l.upper()+'/'+filename[:-4]+'_gt.jpg',image_drawing);
            success = cv2.imwrite('train/'+l.upper()+'/'+filename[:-4]+'_m.jpg',image_mask.astype(np.uint8));
	    # compute SIFT keypoint and feature
	    ## get gray image
            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);
	    ## instantiate and call SIFT extractor
	    sift = cv2.SIFT(nOctaveLayers=4,contrastThreshold=0.04,edgeThreshold=10,sigma=1.2);
	    kp,des = sift.detectAndCompute(gray,None);
	    ## draw SIFT keypoints
	    gray_kp = cv2.drawKeypoints(gray,kp,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
	    success = cv2.imwrite('train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(count)+'_siftkp.jpg',gray_kp);
	    ## prune and draw SIFT keypoints
	    kp_pruned,des_pruned = prune_keypoints(kp,des);
	    gray_kp_pruned = cv2.drawKeypoints(gray,kp_pruned,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
	    success = cv2.imwrite('train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(count)+'_siftkp_pruned.jpg',gray_kp_pruned);
	    ## count number of features per image
	    nof_features['train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(count)+'_siftkp.jpg']=len(kp);
	    nof_features_pruned['train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(count)+'_siftkp_pruned.jpg']=len(kp_pruned);
	    count = count+1


