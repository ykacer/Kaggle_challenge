import json
import cv2
import numpy as np

labels = ['alb','bet','dol','lag','other','shark','yft']

for l in labels:
    print l.upper()+'...'
    files = json.load(open(l+'_labels.json'));
    for f in files:
        filename = f['filename'];
        annotations = f['annotations']
        classe = f['class']
        image = cv2.imread('train/'+l.upper()+'/'+filename,-1);
        image_mask = np.zeros_like(image);
        for a in annotations:
            x = int(a['x'])
            x = x-x*(x<0)
            y = int(a['y'])
            y = y-y*(y<0)
            h = int(a['height'])
            w = int(a['width'])
            c = a['class']
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.rectangle(image_mask,(x,y),(x+w,y+h),(255,255,255),thickness=cv2.cv.CV_FILLED)
        cv2.imwrite('train/'+l.upper()+'/'+filename+'_gt.jpg',image);
        cv2.imwrite('train/'+l.upper()+'/'+filename+'_m.jpg',image_mask.astype(np.uint8));




