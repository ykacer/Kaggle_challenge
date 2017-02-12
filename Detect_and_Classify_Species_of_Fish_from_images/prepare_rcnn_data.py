# -*- coding: utf-8 -*-
import shutil
import os

import json
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import copy
import re
	
import numpy as np

import cv2

labels = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT','NoF']

main_path = 'rcnn/TheNatureConservancy/Kaggle2017/' 
annotation_path = main_path+'/Annotations/'
imagesets_path = main_path+'/ImageSets/'
jpegimages_path = main_path+'/JPEGImages/'

if os.path.isdir(main_path):
    os.system('rm -rf '+main_path)

try:
    os.system('mkdir -p '+main_path)
    os.system('sleep 5')
    os.mkdir(annotation_path)
    os.mkdir(imagesets_path)
    os.mkdir(jpegimages_path)
    os.mkdir(imagesets_path+'/Layout')
except:
    pass

train_set_file = open(imagesets_path+'Layout/train.txt','w')
val_set_file = open(imagesets_path+'Layout/val.txt','w')
trainval_set_file = open(imagesets_path+'Layout/trainval.txt','w')
test_set_file = open(imagesets_path+'Layout/test.txt','w')

folder = 'Kaggle_Fish'
source = {}
source['database'] = 'Detect_and_Classify_Fish'
source['annotation'] = 'Kaggle'
source['image'] = 'flickr'
source['flickrid'] = '0'
owner = {}
owner['flickrid'] = '?'
owner['name'] = 'The Nature Conservancy'


count = 0
for l in labels:
    print l+'...'
    files = json.load(open('annotations/'+l.lower()+'_labels.json'));
    for f in files:
	filename = f['filename'];
	classe = f['class']
        img = cv2.imread('train/'+l+'/'+filename,-1);
        height,width,depth = img.shape
	annotation_dict = {
	'annotation':
	 {
	 	'folder':      folder,
	 	'filename':    filename,
	 	'source':
	 	{
	  		'database':   source['database'],
	  		'annotation': source['annotation'],
	  		'image':      source['image'],
	  		'flickrid':   source['flickrid']
	 	},
	 	'owner':
	 	{
	  		'flickrid':   owner['flickrid'],
	  		'name':       owner['name']
	 	},
	 	'size':
	 	{
			'width':      str(width),
	  		'height':     str(height),
	  		'depth':      str(depth)
	 	},
		'segmented':   '0'
	}
	}

        if count%3 == 0:
            train_set_file.write(filename+'\n')
            trainval_set_file.write(filename+'\n')
        elif count%3 == 1:
            val_set_file.write(filename+'\n')
            trainval_set_file.write(filename+'\n')
        else:
            test_set_file.write(filename+'\n')
        count = count+1;
        if 'annotations' in f.keys():
		annotations = f['annotations']
		objects = []
		for a in annotations:
			x = int(a['x'])
			x = x-x*(x<0)
			y = int(a['y'])
			y = y-y*(y<0)
			h = int(a['height'])
			w = int(a['width'])
			c = a['class']
		        t = 0
		        if (x==1) | (y==1):
		            t = 1
		        o = {'object':
		                {
		                    'name':         'fish',
		                    'pose':         'unknown',
		                    'truncated':    str(t),
		                    'difficult':    '0',
		                    'bndbox':
		                    {
		                        'xmin':     str(x),
		                        'ymin':     str(y),
		                        'xmax':     str(x+w),
		                        'ymax':     str(y+h)
		                    }
		                }
		            }
		        objects.append(o)
		
		annotation_dict['annotation']['objects'] = objects
        annotation_xml = dicttoxml(annotation_dict,root=False,attr_type=False)
        xml_string = parseString(annotation_xml).toprettyxml()
        xml_string = re.sub('<item>','',xml_string)
        xml_string = re.sub('</item>\n','',xml_string)
        xml_string = re.sub('<objects>\n','',xml_string)
        xml_string = re.sub('</objects>','',xml_string)
        xml_file = open(annotation_path+filename[:-3]+'xml','w')
        xml_file.write(xml_string)
        xml_file.close()
        shutil.copyfile('train/'+l+'/'+filename,jpegimages_path+'/'+filename)
       
train_set_file.close();
val_set_file.close();
trainval_set_file.close();
test_set_file.close();


