import cv2
import numpy as np
import os
import pandas as pd
import csv
import cv2
from itertools import izip

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.externals import joblib

def gen_chunks(reader, chunksize=100):
    """
    Chunk generator. Take a CSV `reader` and yield
    `chunksize` sized slices.
    """
    chunk = []
    dummy = reader.next()
    for i, line in enumerate(reader):
        if (i % chunksize == 0 and i > 0):
            yield chunk
            del chunk[:]
        chunk.append(line)
    yield chunk

# encode all possible classes of fish
classes = {}
classes['ALB'] = 1;
classes['BET'] = 2;
classes['DOL'] = 3;
classes['LAG'] = 4;
classes['SHARK'] = 5;
classes['YFT'] = 6;
classes['OTHER'] = 7;

descriptor_type = 'surf'

# csv file containing all vector descriptors of each image, and label of each image
descriptors_file = descriptor_type+'_features.csv'

# number of words (number of clusters during descriptors clustering)
dictionarySize = 1024;
bow = cv2.BOWKMeansTrainer(dictionarySize)

# get number of descriptors to instanciate descriptors array
N = int(os.popen('cat '+descriptors_file+' | wc -l').read())
features = np.zeros((N-1,128));
image_names = []
labels = []

# read descriptors chunk by chunk and fill image name and label of each descriptor
chunk_size = 100;
l = 0;
with open(descriptors_file,'r') as csv_file:
	datareader = csv.reader(csv_file);
	for chunk in gen_chunks(datareader,chunk_size):
		chunk_array = np.asarray(chunk)
		im = chunk_array[:,1]
		lbl = chunk_array[:,-1]
		image_names = image_names+list(im)
		labels = labels+list(lbl)
		features[l:l+len(chunk),:] = chunk_array[:,8:136].astype(np.float)
		l = l + len(chunk)

# gather label of each image
image2label_dict = {}
for im,l in izip(image_names,labels):
	image2label_dict[im] = l

# cluster descriptors
dictionary = bow.cluster(features.astype(np.float32))
joblib.dump((dictionary), 'dico_'+descriptor_type+'_bow.pkl', compress=3)
# instanciate feature extractor structure (with same parameters as those used to create descriptors_file)
# and then perform bag-of-words feature extractor
# sift = cv2.SIFT(nOctaveLayers=4,contrastThreshold=0.02,edgeThreshold=20,sigma=1.2)
if descriptor_type == 'sift':
	descriptor_extractor = cv2.SIFT(nOctaveLayers=4,contrastThreshold=0.02,edgeThreshold=10,sigma=2.1)
else:
	descriptor_extractor = cv2.SURF(200)

#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks=50)   # or pass empty dictionary
#flann = cv2.FlannBasedMatcher(index_params,search_params)
matcher = cv2.BFMatcher(cv2.NORM_L2)

#flann_params = dict(algorithm = 1, trees = 5)      # flann enums are missing, FLANN_INDEX_KDTREE=1
#matcher = cv2.FlannBasedMatcher(flann_params, {}) # need to pass empty dict (#1329)
if descriptor_type == 'sift':
	sift = cv2.DescriptorExtractor_create("SIFT")
	bowExtractor = cv2.BOWImgDescriptorExtractor(sift,matcher)
else:
	surf = cv2.DescriptorExtractor_create("SURF")
	surf.setBool("extended", True)
	bowExtractor = cv2.BOWImgDescriptorExtractor(surf,matcher)

bowExtractor.setVocabulary(dictionary)

set_images = list(set(image_names))
X = np.zeros((len(set_images),dictionarySize))
y = np.zeros((len(set_images),))

for ind,im in enumerate(set_images):
	gray = cv2.cvtColor(cv2.imread(im,-1),cv2.COLOR_BGR2GRAY);
	y[ind] = classes[image2label_dict[im]]
	X[ind,:] = bowExtractor.compute(gray,descriptor_extractor.detect(gray))

# record bow features and labels
bow_features = descriptor_type+'_bow_features.csv'
pd.DataFrame(X).to_csv(bow_features)
bow_labels = descriptor_type+'_bow_labels.csv'
pd.DataFrame(y).to_csv(bow_labels)

# Classification
data = np.concatenate((X,y[:,np.newaxis]),axis=1)
np.random.shuffle(data)

Ntrain = 3000

Xtrain = data[:Ntrain,:-1]
ytrain = data[:Ntrain,-1]

Xtest = data[Ntrain:,:-1]
ytest = data[Ntrain:,-1]

sc = StandardScaler()
Xtrain_r = sc.fit_transform(Xtrain)
Xtest_r = sc.transform(Xtest)

n_cl = 7
n_samples = Xtrain_r.shape[0] 
C = 1.0
weights = 1.0*n_samples / (n_cl * np.bincount(ytrain.astype(np.int64)))
class_weight = dict(zip([1,2,3,4,5,6,7],weights[1:]))
cl = SVC(kernel='rbf',gamma=100.0,decision_function_shape='ovr',probability=True,class_weight=class_weight,random_state=0,verbose=False)
#cl = SVC(kernel='rbf',random_state=0,verbose=False)
cl.fit(Xtrain_r,ytrain)
print np.array_str(metrics.confusion_matrix(ytest,cl.predict(Xtest_r)))

cl = KNeighborsClassifier()
cl.fit(Xtrain_r,ytrain)
print np.array_str(metrics.confusion_matrix(ytest,cl.predict(Xtest_r)))

cl = LogisticRegression(C=100.0,class_weight=class_weight,random_state=0)
cl.fit(Xtrain_r,ytrain)
print np.array_str(metrics.confusion_matrix(ytest,cl.predict(Xtest_r)))

cl = MLPClassifier(hidden_layer_sizes=(2000,500),solver='sgd',activation='relu', batch_size='auto',learning_rate_init=0.1,learning_rate='constant',max_iter=400,random_state=0,tol=0.01,momentum=0.9,nesterovs_momentum=True,early_stopping=True,verbose=True)
cl.fit(Xtrain_r,ytrain)
print np.array_str(metrics.confusion_matrix(ytest,cl.predict(Xtest_r)))

cl = GradientBoostingClassifier(learning_rate=0.01, n_estimators=800,max_depth=24, min_samples_split=160, min_samples_leaf=5, subsample=0.8, max_features=11, random_state=0)
cl.fit(Xtrain_r,ytrain)
print np.array_str(metrics.confusion_matrix(ytest,cl.predict(Xtest_r)))

