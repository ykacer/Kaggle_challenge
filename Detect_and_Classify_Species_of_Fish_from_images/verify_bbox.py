import json
import cv2
import numpy as np
import os

labels = ['alb','bet','dol','lag','other','shark','yft']

for l in labels:
    print l.upper()+'...'
    files = json.load(open(l+'_labels.json'));
    os.mkdir('train/'+l.upper()+'/crop/')
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
	    cv2.imwrite('train/'+l.upper()+'/crop/'+filename[:-4]+'_'+str(count)+'.jpg',crop);
	    count = count+1
            cv2.rectangle(image_drawing,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.rectangle(image_mask,(x,y),(x+w,y+h),(255,255,255),thickness=cv2.cv.CV_FILLED)
        cv2.imwrite('train/'+l.upper()+'/'+filename[:-4]+'_gt.jpg',image_drawing);
        cv2.imwrite('train/'+l.upper()+'/'+filename[:-4]+'_m.jpg',image_mask.astype(np.uint8));

