import os
import random
import numpy as np
from skimage import util,io,transform,feature,color
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.svm import  SVC
from sklearn.metrics import confusion_matrix

path_label='label.txt'
path_train='train//'
path_pos='visage//'
path_neg='negatif//'
path_save='SVM.pkl'
size_fixe=(90,60)

def read_label(filepath):
    f = open(filepath)
    lines = f.readlines()
    label = np.zeros((len(lines), 5))
    for i in range(0, len(lines)):
        label[i, :] = lines[i].split(' ')
    return label

def redimension(ratio, label):
    label_new = np.zeros((len(label), 5))
    label_new[:, 0] = label[:, 0]
    for i in range(0, len(label)):
        hauteur = label[i, 3]
        largeur = label[i, 4]
        if (largeur * ratio < hauteur):
            label_new[i, 1] = label[i, 1] + int((hauteur - largeur * ratio) / 2.0)
            label_new[i, 2] = label[i, 2]
            label_new[i, 3] = int(largeur * ratio)
            label_new[i, 4] = label[i, 4]
        else:
            label_new[i, 1] = label[i, 1]
            label_new[i, 2] = label[i, 2] + int((largeur - hauteur / ratio) / 2.0)
            label_new[i, 3] = label[i, 3]
            label_new[i, 4] = int(hauteur / ratio)
    return label_new

def get_positif(new_label, size,path_train,path_pos):
    count = 0
    for i in range(0, len(new_label)):
        img = util.img_as_float(io.imread(path_train+'%04d.jpg' % int(new_label[i, 0])))
        img_grey=color.rgb2gray(img)
        x, y, hauteur, largeur = int(new_label[i, 1]), int(new_label[i, 2]), int(new_label[i, 3]), int(new_label[i, 4])
        img_visage = img_grey[x:x + hauteur, y:y + largeur]
        img_fixe_taille = transform.resize(img_visage, size)
        io.imsave(path_pos+'%06d.jpg' % count, img_fixe_taille)
        count = count + 1

def calculer_cover_rate(rect1, rect2):
    x1,y1,h1,l1 = rect1[0],rect1[1],rect1[2],rect1[3]
    x2, y2, h2, l2 = rect2[0], rect2[1], rect2[2], rect2[3]
    p1_x, p1_y = np.max((x1, x2)), np.max((y1, y2))
    p2_x, p2_y = np.min((x1 + h1, x2 + h2)), np.min((y1 + l1, y2 + l2))
    if ( p2_y-p1_y==y1 and p2_x-p1_x==x1 )  or ( p2_y-p1_y==y2 and p2_x-p1_x==x2 ) :
        return 1
    AJoin = 0
    if (min(x1 + h1, x2 + h2) >= max(x1, x2)) and (min(y1 + l1, y2 + l2) >= max(y1, y2)):
        AJoin = (p2_x - p1_x) * (p2_y - p1_y)
    AUnion = h1 * l1 + l2 * h2 - AJoin
    return (AJoin / AUnion)

def groupbydicts(label):
    dicts = {}
    for i in label:
        if i[0] not in dicts.keys():
            dicts[i[0]]=[]
        dicts[i[0]].append(i[1:])
    return dicts

def get_negatif(label, size,path_train,path_neg):
    label_dict=groupbydicts(label)
    index= 0
    for img_num in label_dict.keys():
        img = color.rgb2gray(util.img_as_float(io.imread(path_train+'%04d.jpg' % int(img_num))))
        img_h, img_l = np.shape(img)[0], np.shape(img)[1]
        numberOfFace = len(label_dict[img_num])
        for visage_num in range(0,len(label_dict[img_num])):
             maxIteration,count =  0,0
             visage = label_dict[img_num][visage_num]
             h_window, l_window = int(visage[2]), int(visage[3])
             rest=10%numberOfFace
             thresold = 10//numberOfFace if visage_num < len(label_dict[img_num]) - rest else 10//numberOfFace + 1
             while count < thresold and maxIteration < 100000:
                        x1 = random.randint(0, img_h - h_window)
                        y1 = random.randint(0, img_l - l_window)
                        flag = True
                        for visageCompared  in label_dict[img_num]:
                             x, y, h, l = int(visageCompared[0]), int(visageCompared[1]), int( visageCompared[2]), int(visageCompared[3])
                             if calculer_cover_rate((x1, y1,h_window,l_window),( x, y, h,l))>0.5:
                                     flag = False
                                     break
                        if flag:
                                    img_visage = img[x1:x1 + h_window, y1:y1 + l_window]
                                    img_resize = transform.resize(img_visage, size)
                                    io.imsave(path_neg+'%06d.jpg' % index,img_resize)
                                    count = count + 1
                                    index= index +1
                        maxIteration = maxIteration + 1

def get_data(path,size):
    images = os.listdir(path)
    images_data = np.zeros((len(images), size[0], size[1]))
    for image, index in zip(images, range(len(images))):
        im = util.img_as_float(io.imread(os.path.join(path, image)))
        images_data[index, :, :] = im
    return (len(images_data), images_data)

def get_trainingdata_HOG(path_pos,path_neg,size):
    pos_length, pos_images = get_data(path_pos,size)  # Get positive image data
    neg_length, neg_images = get_data(path_neg,size)  # Get negative image data
    images = np.concatenate((pos_images,neg_images))
    labels=np.concatenate((np.repeat(1,pos_length),np.repeat(0,neg_length)))
    img_HOG=[]
    for img in images:
        img_HOG.append(feature.hog(img))
    img_HOG, labels = shuffle(img_HOG,labels)
    joblib.dump(img_HOG,"img_HOG.pkl")
    joblib.dump(labels,"labels_HOG.pkl")
    return img_HOG,labels

def train_generate_modelSVC():
    data = joblib.load("img_HOG.pkl")
    label=joblib.load("labels_HOG.pkl")
    clf=SVC(C=1,kernel="linear")
    fp=1000
    while fp>0:
        clf.fit(data,label)
        prediction=clf.predict(data)
        number=len(data)
        C = confusion_matrix(label, prediction)
        print(C)
        fp=C[0][1]
        for index,i,j in zip(range(number),label,prediction):
            if i==0 and j==1:
                data=np.concatenate((data,[data[index]]))
                label=np.concatenate((label,[label[index]]))
    joblib.dump(clf,'SVM.pkl')

