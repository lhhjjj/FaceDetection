from train import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def plot_histgrammeof_hauteurs():
            label = read_label(path_label)
            ratio = round(np.mean(label[:, 3] / label[:, 4]), 1)
            new_label = redimension(ratio, label)
            plt.hist(new_label[:,3])
            plt.xlabel('Hauteur des visages')
            plt.ylabel('Frequence')
            plt.title("Distribution d'hauteurs de visages")
            plt.show()

def get_trainingdata_pixel(path_pos, path_neg, size):
    pos_length, pos_images = get_data(path_pos, size)  # Get positive image data
    neg_length, neg_images = get_data(path_neg, size)  # Get negative image data
    images = np.concatenate((pos_images, neg_images))
    labels = np.concatenate((np.repeat(1, pos_length), np.repeat(0, neg_length)))
    img_pixel = np.zeros((len(images), size[0] * size[1]))
    for i in range(0, len(images)):
        img_pixel[i, :] = np.ravel(images[i, :, :])
    img_pixel, labels = shuffle(img_pixel, labels)
    return img_pixel, labels

def to_daisy(img):
    img_daisy = feature.daisy(img, step=10, radius=10, rings=3, histograms=8, orientations=8)
    return np.reshape(img_daisy, (img_daisy.shape[0] * img_daisy.shape[1], img_daisy.shape[2]), order='F')

def get_trainingdata_daisy(path_pos, path_neg, size):
    pos_length, pos_images = get_data(path_pos, size)  # Get positive image data
    neg_length, neg_images = get_data(path_neg, size)  # Get negative image data
    images = np.concatenate((pos_images, neg_images))
    labels = np.concatenate((np.repeat(1, pos_length), np.repeat(0, neg_length)))
    img_num = np.zeros(len(images))
    imgs = np.empty((0, 200))
    for i in range(0, len(images)):
        img_daisy = to_daisy(images[i, :, :])
        imgs = np.concatenate((imgs, img_daisy))
        img_num[i] = len(imgs)
    clf = KMeans(n_clusters=20)
    clf.fit(imgs)
    label = clf.predict(imgs)
    img_hist = np.zeros((len(images), 20))
    for i in range(0, len(images)):
        i_start = int(img_num[i - 1]) if i > 0 else 0
        i_end = int(img_num[i])
        img_hist[i, :] = np.bincount(label[i_start:i_end], minlength=20)
        img_hist[i, :] /= np.sum(img_hist[i, :])
    img_hist, labels = shuffle(img_hist, labels)
    joblib.dump(img_hist, "img_hist.pkl")
    joblib.dump(labels, "img_labels.pkl")
    return img_hist, labels

def ROC_daisy_hog_pixel():
    lw = 2
    plt.figure(figsize=(5, 5))
    hog_score, hog_pred = joblib.load('hog_sco.pkl'), joblib.load( 'hog_pred.pkl')
    pixel_score, pixel_pred = joblib.load('pixel_sco.pkl'), joblib.load('pixel_pred.pkl')
    daisy_score, daisy_pred = joblib.load('daisy_sco.pkl'), joblib.load( 'daisy_pred.pkl')
    fpr1, tpr1, threshold1 = roc_curve(hog_pred, hog_score)
    roc_auc1 = auc(fpr1, tpr1)
    fpr2, tpr2, threshold2 = roc_curve(pixel_pred, pixel_score)
    roc_auc2 = auc(fpr2, tpr2)
    fpr3, tpr3, threshold3 = roc_curve(daisy_pred, daisy_score)
    roc_auc3 = auc(fpr3, tpr3)
    plt.plot(fpr1, tpr1, color='darkorange', lw=lw, label='HOG (area = %0.4f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='darkred', lw=lw, label='Pixel (area = %0.3f)' % roc_auc2)
    plt.plot(fpr3, tpr3, color='darkgreen', lw=lw, label='Daisy(area = %0.3f)' % roc_auc3)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ')
    plt.legend(loc="lower right")
    plt.show()

def CrossValidation(data,label,clf,k):
    data,label=np.asarray(data),np.asarray(label)
    kf = KFold(n_splits=k,shuffle=False)
    mean_err = 0
    count=1
    for train_index,test_index in kf.split(data):
        count += 1
        train_data,test_data= data[train_index],data[test_index]
        train_label,test_label=label[train_index],label[test_index]
        clf.fit(train_data,train_label)
        predict_accuracy = clf.score(test_data,test_label)
        mean_err += 1 - predict_accuracy
    mean_err /= k
    return mean_err

def AdaBoostClf(data, label):
    clf = AdaBoostClassifier()
    taux_error = CrossValidation(data, label, clf, 5)
    print("Le taux d'erreur pour AdaBoost : %.4f" % taux_error)

def RandomForestClf(data, label):
    clf = RandomForestClassifier()
    taux_error = CrossValidation(data, label, clf, 5)
    print("Le taux d'erreur pour RandomForest : %.4f" % taux_error)

def Choix_of_paras_SVM(data, label,noyau):
    Cs = [0.01, 0.1, 1, 10, 50, 100]
    errs = np.zeros(len(Cs))
    for c in range(0, len(Cs)):
        clf = SVC(C=Cs[c], kernel=noyau)
        errs[c] = CrossValidation(data, label, clf, 5)
    joblib.dump(errs, 'errs_%s.pkl' % noyau)
    return errs

def Courbe_errs_SVCs():
    errs_linear, errs_rbf, errs_poly = joblib.load( "errs_linear.pkl"), joblib.load("errs_rbf.pkl"), joblib.load("errs_poly.pkl")
    plt.figure(figsize=(5, 5))
    Cs = [0.01, 0.1, 1, 10, 50, 100]
    x = [1, 2, 3, 4, 5, 6]
    plt.plot(x, errs_linear, color='darkorange', label='Linear', marker='o', markersize=5)
    plt.plot(x, errs_rbf, color='darkred', label='RBF', marker='o', markersize=5)
    plt.plot(x, errs_poly, color='darkblue', label='Poly', marker='o', markersize=5)
    plt.ylim([0.0, 0.13])
    plt.xticks(x, Cs)
    plt.xlabel('Les valeurs de C')
    plt.ylabel("Taux d'erreur")
    plt.title('La performance des SVCs avec les noyaux et C differents ')
    plt.legend(loc="higher right")
    plt.show()