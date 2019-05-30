from sklearn.svm import SVC
from sklearn.externals import joblib

from skimage import data, io, util, color
from skimage.transform import pyramid_gaussian
from skimage.feature import hog
import pickle
import numpy as np

import math

HEIGHT = 90.0
WIDTH = 60.0
SCALE = math.sqrt(2)

def apply_window(image, model, preprocess, hstep=1, wstep=1):
    if image.shape[0] < int(HEIGHT) or image.shape[1] < int(WIDTH):
        return []

    matched_points = list()
    scores = list()

    positions = list()
    preprocessed_datas = list()
    
    print("Preprocessing: height step:{}, width step:{}".format(hstep, wstep))

    for i in range(0, image.shape[1] - int(WIDTH), wstep):
        for j in range(0, image.shape[0] - int(HEIGHT), hstep):
            sub_image = image[j:j+int(HEIGHT), i:i+int(WIDTH)] 

            processed_data = preprocess(sub_image)

            preprocessed_datas.append(processed_data)
            positions.append((j, i))

    print("Preprocessing finished")
    predict_results = model.predict(preprocessed_datas)

    predict_scores = model.decision_function(preprocessed_datas)

    for position, result, score in zip(positions, predict_results, predict_scores):
        if result == 1 and score >= 2.1:
            matched_points.append(position)
            scores.append(score)

    return (matched_points, scores)


def get_image_pyramid(image, layer=-1, downscale=math.sqrt(2)):
    pyramid_gaussian(image_f, max_layer=5, downscale=math.sqrt(2))


def read_image(filename):
    return util.img_as_float(color.rgb2gray(io.imread(filename)))


def hog_feature(image):
    return hog(image)


def get_cover_rate(rect1, rect2):
    x1, y1, h1, l1 = (int(rect1[0]), int(rect1[1]), int(rect1[2]), int(rect1[3]))
    x2, y2, h2, l2 = (int(rect2[0]), int(rect2[1]), int(rect2[2]), int(rect2[3]))
    p1_x, p1_y = np.max((x1, x2)), np.max((y1, y2))
    p2_x, p2_y = np.min((x1 + h1, x2 + h2)), np.min((y1 + l1, y2 + l2))
    
    if ( p2_y-p1_y==y1 and p2_x-p1_x==x1 )  or ( p2_y-p1_y==y2 and p2_x-p1_x==x2 ) :
        return 1
        
    a_join = 0
    if (min(x1 + h1, x2 + h2) >= max(x1, x2)) and (min(y1 + l1, y2 + l2) >= max(y1, y2)):
        a_join = (p2_x - p1_x) * (p2_y - p1_y)
    a_union = h1 * l1 + l2 * h2 - a_join
    return a_join / a_union


def critere(label1, label2):
    if label1[3] > label2[3]:
        return 0.5 / pow(float(label1[3]) / label2[3], 2)
    else:
        return 0.5 / pow(float(label2[3]) / label1[3], 2)

def suppresion_non_maxima_meme_couche(positions,scores,seuil_of_recouvrement):
    i_base = 0
    while i_base < len(positions):
        i_comp = i_base + 1
        while i_comp < len(positions):
            cover_rate = get_cover_rate((positions[i_base][0], positions[i_base][1], HEIGHT, WIDTH),(positions[i_comp][0], positions[i_comp][1], HEIGHT, WIDTH))
            if cover_rate > seuil_of_recouvrement:
                if scores[i_comp] > scores[i_base]:
                    del positions[i_base]
                    del scores[i_base]
                    i_base -= 1
                    break
                elif scores[i_comp] < scores[i_base]:
                    del positions[i_comp]
                    del scores[i_comp]
                    i_comp -= 1
            i_comp += 1
        i_base += 1

def suppresion_non_maxima_parmi_couches(labels):
    i_base = 0
    while i_base < len(labels):
        i_comp = i_base + 1
        while i_comp < len(labels):
            cover_rate = get_cover_rate(labels[i_base][1:-1], labels[i_comp][1:-1])
            if cover_rate > critere(labels[i_base], labels[i_comp]):
                if labels[i_comp][-1] > labels[i_base][-1]:
                    del labels[i_base]
                    i_base -= 1
                    break
                elif labels[i_comp][-1] < labels[i_base][-1]:
                    del labels[i_comp]
                    i_comp -= 1
            i_comp += 1
        i_base += 1

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Test an image.')
    parser.add_argument('image', type=int, default=-1, nargs='?', help='index of image')

    args = parser.parse_args()

    if args.image <= 0:  # Detect all
        image_indexs = [i for i in range(1, 500+1)]
    else:               # Detect single image
        image_indexs = [args.image]

    model = joblib.load("SVM.pkl")

    for image_index in image_indexs:
        image_f = read_image("test/%04d.jpg"%image_index)

        images_pyramid = tuple(pyramid_gaussian(image_f, max_layer=5, downscale=SCALE))

        labels = list()

        print("=========================================================")
        print("%04d"%image_index)
        print("=========================================================")
        
        for image_pyramid, i in zip(images_pyramid, range(len(images_pyramid))):
            if image_pyramid.shape[0] > HEIGHT and image_pyramid.shape[1] > WIDTH:
                positions, scores = apply_window(image_pyramid, model, hog_feature, int(round(max(4 / pow(SCALE, i), 1))), int(round(max(3 / pow(SCALE, i), 1))))
                
                suppresion_non_maxima_meme_couche(positions, scores, 0.5)
                
                ratio = math.pow(SCALE, i)

                for position, score in zip(positions, scores):
                    label = (image_index, int(position[0]*ratio), int(position[1]*ratio), int(HEIGHT*ratio), int(WIDTH*ratio), score)
                    labels.append(label)
                print("Size ok {}".format(image_pyramid.shape))
            else:
                print("Size not ok {}".format(image_pyramid.shape))

        # Remove duplicated between scales
        suppresion_non_maxima_parmi_couches(labels)
        if args.image <= 0:
               file_decision = open("decision.txt", "a")
        else :
              f = open("test-%04d.txt" % image_index, "a")
        for label in labels:
            # Sequentiellement
            if args.image <= 0:
                file_decision.write("%03d %d %d %d %d %.2f\n"%label)
                io.imsave(("%04d-{}-{}-{}-{}-{}.jpg" % image_index).format(label[3], label[4], label[1], label[2], label[5]), image_f[label[1]:label[1] + label[3], label[2]:label[2] + label[4]])
            else:
                io.imsave(("%04d-{}-{}-{}-{}-{}.jpg"%image_index).format(label[3], label[4], label[1], label[2], label[5]), image_f[label[1]:label[1]+label[3], label[2]:label[2]+label[4]])
                f.write("%03d %d %d %d %d %.2f\n"%label)
        if args.image <= 0:
            file_decision.close()
        else :
             f.close()

