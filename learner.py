#!/usr/bin/python
'''train the classifier'''

from skimage.io import imread, imshow, imsave
from skimage import img_as_float
from skimage.feature import hog
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
from skimage.transform import rescale

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV

import sys
import glob
import numpy as np
import pylab
import pickle
import random
import os

def unsharp(image, unsharp_blurr_size=8, unsharp_strength=0.8):
    '''Sharpen the image using the unsharp method'''
    blurred = gaussian(image, unsharp_blurr_size)
    highpass = image - unsharp_strength*blurred
    sharp = image + highpass
    return sharp/(sharp.max() - sharp.min())

def unsharp_and_local_equalize(image, unsharp_strength=0.8, unsharp_blurr_size=5, clip_limit=0.05):
    '''Unsharp and local equalize the image'''
    image = unsharp(image, unsharp_blurr_size,  unsharp_strength)
    image = equalize_adapthist(image, clip_limit=clip_limit)
    return image

def get_images_at_scale(database, scale, target):
    '''Return list of images taken at a particular "scale", scale argument can be "dewar", "lid" or "puck"'''
    images_at_scale = []
    for imagename in database:
        if type(database[imagename]) == dict and database[imagename]['scale'] == scale and database[imagename].has_key('%s_present' % target) and database[imagename].has_key('%s_not_present' % target) and (len(database[imagename]['%s_present' % target]) > 0 or len(database[imagename]['%s_not_present' % target]) > 0):
            images_at_scale.append(imagename)
    return images_at_scale

def get_closest_click(click, clicks):
    '''The function will return the closest coordinates of the "click" point in the list of coordinates "clicks"'''
    nclicks= np.array(clicks)
    nclick = np.array(click)
    a = np.apply_along_axis(np.linalg.norm, 1, nclicks - nclick)
    return clicks[np.argmin(a)]

def get_clicks(database, imagename, nnegative=300, boundary=25, nmindistance=45, nmaxdistance=150, target='puck'):
    '''The function will return arrays of coordinates of positive and negative examples for given image. 
       The function will also generate random coordinates of negative examples, conditioned on sufficient distance of the positive examples ones'''
    present = list(set(database[imagename]['%s_present' % target]))
    not_present = list(set(database[imagename]['%s_not_present' % target]))
    negative = list(set(database[imagename]['%s_negative' % target])) if database[imagename].has_key('%s_negative' % target) else []
    
    positive_clicks = present + not_present
    
    while len(negative) < nnegative:
        x_index = random.randint(0+boundary, 704-boundary)
        y_index = random.randint(0+boundary, 576-boundary)
        cc = get_closest_click((x_index, y_index), positive_clicks)
        if np.linalg.norm(np.array([x_index, y_index]) - np.array(cc)) > nmindistance and np.linalg.norm(np.array([x_index, y_index]) - np.array(cc)) < nmaxdistance:
            negative.append((x_index, y_index))
    
    return present, not_present, negative

def main():
    import optparse
    parser = optparse.OptionParser()
    
    parser.add_option('-d', '--database', default='images_scales.pickle', type=str, help='database with image information')
    parser.add_option('-s', '--scale', default='dewar', type=str, help='at what scale are we learning')
    parser.add_option('-t', '--target', default='puck', type=str, help='what are we concerned with')
    parser.add_option('-p', '--window', default=119, type=int, help='detection window size')
    parser.add_option('-c', '--components', default=105, type=int, help='number of components for PCA to retain')
    parser.add_option('-n', '--nimages', default=30, type=int, help='number of turbulent images to take for each configuration')
    parser.add_option('-N', '--nnegative', default=30, type=int, help='number of negative clicks for each configuration')
    parser.add_option('-I', '--nmindistance', default=40, type=int, help='minimum distance of negative examples from positive ones')
    parser.add_option('-A', '--nmaxdistance', default=145, type=int, help='maximum distance of negative examples from positive ones')
    parser.add_option('-a', '--augument', dest='augument', action='store_true', default=False, help='Augument the dataset with mirrored and inverted examples.')
    parser.add_option('-C', '--C_parameter', default=5, type=float, help="C support vector parameter")
    options, args = parser.parse_args()

    database = pickle.load(open(options.database))
    
    images_at_scale = get_images_at_scale(database, options.scale, options.target)
    print 'len(images_at_scale)', len(images_at_scale)
    
    X_present = []
    X_not_present = []
    X_negative = []
    
    segment_size = options.window/2 
    if options.rescale != None:
        segment_size *= options.rescale
        segment_size = int(round(segment_size))
    
    for number, item in enumerate(images_at_scale):
        if 'zoom' in item and not 'zoom5' in item:
            print 'not taking into account', item, 'please check'
            continue
        print 'item', number, item
        directory = os.path.dirname(item)
        images_in_dir = os.listdir(directory)
        selected_images = []
        if options.nimages == -1:
            imagename = item
            selected_images.append(os.path.basename(imagename))
        else:
            while len(selected_images) < options.nimages or len(selected_images) > len(images_in_dir) - 3:
                imagename = random.choice(images_in_dir)
                if 'merged' not in imagename and imagename not in selected_images:
                    selected_images.append(imagename)
        
        present, not_present, negative = get_clicks(database, item, nnegative=options.nnegative, boundary=1, nmindistance=options.nmindistance, nmaxdistance=options.nmaxdistance, target=options.target)
        
        pucks_located = False
        for imagename in selected_images:
            
            image = imread('%s' % (os.path.join(directory, imagename),), as_grey=True)
            image = unsharp_and_local_equalize(image)
            
            e = 0
            for k, Clicks, Example_list in ((1, present, X_present), (2, not_present, X_not_present), (0, negative, X_negative)):
                for x, y in np.array(Clicks).astype(int):                    
                    example = image[y-segment_size:y+segment_size + 1, x-segment_size:x+segment_size + 1]
                    if example.shape[0] * example.shape[1] == (2*segment_size + 1)**2:
                        Example_list += [example.ravel()]
                        if options.augument:
                            example_hmirror = image[y-segment_size:y+segment_size + 1, x-segment_size:x+segment_size + 1][:,::-1]
                            example_vmirror = image[y-segment_size:y+segment_size + 1, x-segment_size:x+segment_size + 1][::-1,:]
                            example_inverse = image[y-segment_size:y+segment_size + 1, x-segment_size:x+segment_size + 1].T
                            Example_list += [example_hmirror.ravel(), example_vmirror.ravel(), example_inverse.ravel()]
                        
    X_present = np.array(X_present).T
    X_not_present = np.array(X_not_present).T
    X_negative = np.array(X_negative).T

    print 'X_present.shape', X_present.shape
    print 'X_not_present.shape', X_not_present.shape
    print 'X_negative.shape', X_negative.shape
    
    labels_negative = [0] * X_negative.shape[1]
    labels_present = [1] * X_present.shape[1]
    labels_not_present = [2] * X_not_present.shape[1]
    
    features = np.hstack([X_present, X_not_present, X_negative])
    print 'features.shape', features.shape
    
    pca = PCA(n_components=options.components)
    pca.fit(features.T)
    print 'pca.components', pca.components_.shape
    
    features = pca.transform(features.T)
    print 'transformed features.shape', features.shape
    labels = np.hstack([labels_present, labels_not_present, labels_negative])
    
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, test_size=0.2)
    print 'labels_train', labels_train
    print 'labels_test', labels_test
    
    clf = SVC(C=options.C_parameter, kernel='rbf')
    clf.fit(features_train, labels_train)
    
    filename = 'svc_scale_%s_target_%s_components_%d_window_%d_mind_%d_maxnd_%d_nimages_%d_nnegative_%d_C_%.2f.pkl' % (options.scale, options.target, options.components, options.window, options.nmindistance, options.nmaxdistance, options.nimages, options.nnegative, options.C_parameter)
    
    f = open(filename, 'w')
    pickle.dump({'classifier': clf, 'feature_pca_extractor': pca, 'n_components': options.components, 'window': options.window}, f)
    f.close()
    
    print 'train score', clf.score(features_train, labels_train)
    print 'test score', clf.score(features_test, labels_test)
    print 'cofusion_matrix'
    print confusion_matrix(labels_test, clf.predict(features_test))
    print 'classification_report'
    print classification_report(labels_test, clf.predict(features_test))
    
    sys.exit(0)
    
    
if __name__ == '__main__':
    main()
