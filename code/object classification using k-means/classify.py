import cv2
import numpy as np
import os
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering as Ac
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

path="Dataset/"
clusters=50
entries=os.listdir(path)
class_count=np.array([])
desc = np.array([])
np.random.seed(0)

descriptors = np.zeros(128)
count=0
flag=True
for item in entries:
    count+=1
    img_path=path+str(item)+"/"
    image_array=os.listdir(img_path)
    for pic in image_array:
        new_path=img_path+str(pic)
        print new_path
        img=cv2.imread(new_path,0)
        sift=cv2.SIFT()
        kp,des=sift.detectAndCompute(img,None)
        des=normalize(des,axis=1)
##        print des
        descriptors = np.vstack((descriptors,des))
        if(flag==False):
            desc=np.delete(desc,(0),axis=0)
            flag=True
            class_count=np.append(class_count,count)
        else:  
            class_count=np.append(class_count,count)

descriptors=np.delete(descriptors,(0),axis=0)
##print "shape is: ",descriptors.shape           
desc = np.array(descriptors)
desc = np.float32(desc)

print "Implementing Clustering: "
##h_cluster=Ac(n_clusters=clusters, affinity='euclidean',linkage='average')
h_cluster=cluster.KMeans(n_clusters=clusters)

h_cluster.fit(desc)

Histogram=np.array([])

print "Handling every image now: "
#training each image again
for item in entries:
    count+=1
    img_path=path+str(item)+"/"
    image_array=os.listdir(img_path)
    for pic in image_array:
        LabelHistogram=np.zeros(clusters)
        new_path=img_path+str(pic)
        print new_path
        img=cv2.imread(new_path,0)
        sift=cv2.SIFT()
        kp,des=sift.detectAndCompute(img,None)
        norm=normalize(des,axis=1)
        LabelOfEveryDescriptor=h_cluster.predict(norm)
        
        for i in range(0,len(LabelOfEveryDescriptor)):
            LabelHistogram[LabelOfEveryDescriptor[i]]+=1
        Histogram=np.append(Histogram,LabelHistogram)

Histogram=np.reshape(Histogram,(len(class_count),clusters))

print "Implementing Naive bayes: "
clf = BernoulliNB(alpha=0.0001, class_prior=None, fit_prior=True)
clf.fit(Histogram,class_count)


print "Accuracy of the training data: ",accuracy_score(class_count,clf.predict(Histogram))

print "Testing phase is: "
#testing of the classifier
test_path="Test/"
entries=os.listdir(test_path)
predictions = []
##y_test=[2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,1,1,3,1,1,1,1,1,1,1,1,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
y_test=np.array([2,1,4,3])
for pic in entries:
    LabelHistogram=np.zeros(clusters)
    new_path=test_path+str(pic)
    print new_path
    img=cv2.imread(new_path,0)
    sift=cv2.SIFT()
    kp,des=sift.detectAndCompute(img,None)
    norm=normalize(des,axis=1)
    label_Test=h_cluster.predict(norm)
    
    for i in range(0,len(label_Test)):
            LabelHistogram[label_Test[i]]+=1  
            
    a=clf.predict(LabelHistogram)
    predictions.append(a)
    if(a==1):
        print "Detected object is: Camera"
    elif(a==2):
        print "Detected object is: Dollar Bill"
    elif(a==3):
        print "Detceted object is: Watch"
    else:
        print "Detected object is: Motorcycle"
        
predictions=np.array(predictions)        
print confusion_matrix(y_test,predictions)
##print classification_report(y_test,predictions)
print 'Accuracy at %0.3f' %accuracy_score(y_test,predictions)




        





