#!/usr/bin/env python

'''
Outline of solving the task of determining presence of pins in the CATS dewar

A) Detector
1. generate positive and negative examples: collection and labeling of images
   -- record_the_clicks.py; show_the_clicks.py
2. decide on features, train and decide which classifier to use
   -- learner.py

B)
1. Optical image aquisition: image 
2. Global image treatment: e.g. equalization: image
3. Slicing the image and applying the Detector on each subwindow: array of positive cases (box centers + sizes)
4. Non-maxima suppression: thinned array of positive cases (box centers + sizes)

'''

from multiprocessing import Process, Queue, cpu_count
import numexpr as ne

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow, imsave
from skimage.util import view_as_windows
from skimage import img_as_float 
from skimage.transform import rescale

from learner import unsharp_and_local_equalize

import optparse
import pickle
import time

from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import scipy.ndimage as nd

import math
import sys

puck_angles = {1: 85, 2: -145 , 3: -30, 4: -30, 5: 85, 6: -145, 7: -145, 8: -30, 9: 85}
lid_angles = {'lid1': 0, 'lid2': 120, 'lid3': -120}
lid_order = {'lid1': 1, 'lid2': 2, 'lid3': 3}
puck_order = {'puck1': 1, 'puck2': 2, 'puck3': 3}

def absolute_difference(a):
    return abs(a[0] - a[-1])

def get_lid_indices(lid_centers):
    indices = {}
    
    distances = distance_matrix(lid_centers, lid_centers)
    
    distances = distances[distances!=0].reshape((distances.shape[0], distances.shape[1]-1))
    
    index_lid1 = np.argmin(np.apply_along_axis(absolute_difference, 1, distances))
    
    indices['lid1'] = index_lid1
    
    lid2_and_lid3 = np.ma.array(lid_centers, mask=False)
    
    lid2_and_lid3.mask[index_lid1, :] = True
    
    lid2_and_lid3 = lid2_and_lid3[~lid2_and_lid3.mask].reshape((2, 2))
    
    angles = []
    
    lid2_and_lid3 -= lid_centers[index_lid1, :]
    
    for point in lid2_and_lid3:
        angles.append(np.arcsin(point[1]/np.linalg.norm(point)))
    
    indices['lid2'] = np.argmin(angles)
    indices['lid3'] = np.argmax(angles)
    if indices['lid1'] == 0:
        indices['lid2'] += 1
        indices['lid3'] += 1
    elif indices['lid1'] == 1:
        if indices['lid3'] > indices['lid2']:
            indices['lid3'] = 2
        else:
            indices['lid2'] = 2
    
    reverse_index = {}
    for key in indices:
        reverse_index[indices[key]] = key
    return reverse_index, indices

def rotate(a, unit='radians'):
    if unit != 'radians':
        a = math.radians(a)
        
    r = np.array([[ math.cos(a), -math.sin(a), 0.], 
                  [ math.sin(a),  math.cos(a), 0.], 
                  [          0.,           0., 1.]])
    return r  

def scale(f):
    return np.diag([f, f, 1.])

def shift(displacement):
    s = np.array([[1., 0., displacement[0]],
                  [0., 1., displacement[1]],
                  [0., 0.,              1.]])
    return s


def get_puck_indices(puck_centers, lid, puck_window):
    center = np.mean(puck_centers, axis=0)
    puck_centers -= center
    points = np.hstack((puck_centers, np.ones((puck_centers.shape[0], 1))))
    rotation_matrix = rotate(lid_angles[lid], unit='degrees')
    rotated = np.array([np.dot(rotation_matrix, point.T) for point in points])
    cannonical = rotated.reshape((3,3))[:,:2]
    indices = {}
    index_puck1 = np.argmax(cannonical.sum(axis=1))
    indices['puck1'] = {'index': index_puck1}
    puck2_and_puck3 = np.ma.array(cannonical, mask=False)
    puck2_and_puck3.mask[index_puck1, :] = True
    puck2_and_puck3 = puck2_and_puck3[~puck2_and_puck3.mask].reshape((2, 2))
    angles = []
    for point in puck2_and_puck3:
        angles.append(np.arcsin(point[1]/np.linalg.norm(point)))
    indices['puck3'] = {'index': np.argmin(angles)}
    indices['puck2'] = {'index': np.argmax(angles)}
    if indices['puck1']['index'] == 0:
        indices['puck2']['index'] += 1
        indices['puck3']['index'] += 1
    elif indices['puck1']['index'] == 1:
        if indices['puck2']['index'] > indices['puck3']['index']:
            indices['puck2']['index'] = 2
        else:
            indices['puck3']['index'] = 2
    reverse_index = {}
    for key in indices:
        indices[key]['center'] = tuple(map(int, puck_centers[indices[key]['index'], :] + center - puck_window/2))
        indices[key]['puck_global'] = puck_order[key] + 3 * (lid_order[lid] - 1)
        indices[key]['lid'] = lid
        indices[key]['lid_order'] = lid_order[lid]
        reverse_index[indices[key]['index']] = key
    return reverse_index, indices

def get_pin_indices():
    return

def norm(a):
    return np.linalg.norm(a)

def determine_centers_corresponding_to_present_objects(centers, present, not_present, distance=10):
    present_indices = []
    present_locations = []
    weights = []
    if len(present) != 0:
        for k, center in enumerate(centers):
            n_present = sum(np.apply_along_axis(norm, 1, (present - center)) < distance )
            if len(not_present) > 0:
                n_not_present = sum(np.apply_along_axis(norm, 1, (not_present - center)) < distance )
            else:
                n_not_present = 0 
            if np.any( np.apply_along_axis(norm, 1, (present - center)) < distance ) and n_present > n_not_present:
                present_indices.append(k)
                present_locations.append(center)
                weights.append(np.sum(np.apply_along_axis(norm, 1, (present - center)) < distance))

    return present_indices, np.array(present_locations), np.array(weights)

def get_pin_ideal_coordinates(max_distance=5.1470714977808507):
    s = 0.23314228304723908 * max_distance
    l = 0.50514161326901807 * max_distance
    inner_angles = np.arange(0, 2*np.pi, 2*np.pi/5)
    outer_angles = np.arange(0, 2*np.pi, 2*np.pi/11)
    inner_coordinates = s*np.vstack((np.sin(inner_angles), np.cos(inner_angles))).T
    outer_coordinates = l*np.vstack((np.sin(outer_angles), np.cos(outer_angles))).T
    coordinates = np.vstack((inner_coordinates, outer_coordinates))
    coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    r = rotate(np.pi)
    coordinates = np.array([np.dot(r, point.T) for point in coordinates])
    return coordinates[:, :-1]

def get_puck_ideal_coordinates(max_distance=10):
    return

def get_lid_ideal_coordinates(max_distance=9):
    return

def get_scale_and_center_from_pin_positions(pin_positions):
    center = pin_positions.mean(axis=0)
    coordinates = pin_positions - center
    
    maximum = np.apply_along_axis(norm, 1, coordinates).max()
    priemer = np.apply_along_axis(norm, 1, coordinates).mean()
    minimum = np.apply_along_axis(norm, 1, coordinates).min()
    d = distance_matrix(coordinates, coordinates)
    
    return d.max(), maximum, priemer, minimum, center
    
def rotate_and_scale_and_shift_ideal_pin_coordinates(angle, scale_factor, center):
    coordinates = get_pin_ideal_coordinates(max_distance=scale_factor)
    coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    r = rotate(angle, unit='degrees')
    m = shift(center)
    coordinates = np.array([np.dot(r, point.T) for point in coordinates])
    coordinates = np.array([np.dot(m, point.T) for point in coordinates])
    return coordinates[:, :-1]

def _predict(clf, feature, k, output):
    p = clf.predict(feature)
    output.put((k, p))

def detect_object(image, segment_size, step_size, classifier):
    pw = view_as_windows(image, segment_size, step_size)
    original_shape = pw.shape
    pw = pw.reshape((pw.shape[0]*pw.shape[1], pw.shape[2]*pw.shape[3]))
    
    features = classifier['feature_pca_extractor'].transform(pw)
    clf = classifier['classifier']
    n_cpu = cpu_count()
    
    features_split = np.array_split(features, n_cpu)
    output = Queue()
    processes = []

    for k, feature in enumerate(features_split):
        processes.append(Process(target=_predict, args=(clf, feature, k, output)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    results = [output.get() for p in processes]
    results.sort(key=lambda x: x[0])
    predict = np.array([])
    for result in results:
        predict = np.hstack([predict, result[1]]) if predict.size else result[1]
  
    pm = predict.reshape(original_shape[:2])
    present = np.argwhere(pm==1) * step_size
    not_present = np.argwhere(pm==2) * step_size
    
    return present, not_present

def get_subwindow(image, segment_location, segment_size):
    location = tuple(map(int, segment_location))
    c, r = location
    before_y = int(c)
    after_y = int(c + segment_size)
    before_x = int(r)
    after_x = int(r + segment_size)
    subwindow = image[before_y: after_y, before_x: after_x]
    return subwindow

def get_lid_centers(puck_centers_shifted):
    lids_kmeans = KMeans(n_clusters=3)
    lids_kmeans.fit(puck_centers_shifted)
    return lids_kmeans.cluster_centers_, lids_kmeans

def get_average_lid_distance(lid_centers):
    return distance_matrix(lid_centers, lid_centers).sum()/6

def get_puck_information(puck_centers_shifted, puck_predictions, lid_indices_reverse_index, puck_segment_size):
    puck_information = []
    for k in range(3):
        lid = lid_indices_reverse_index[k]
        puck_reverse_index, puck_indices = get_puck_indices(puck_centers_shifted[puck_predictions==k], lid, puck_segment_size)
        puck_information.append((puck_reverse_index, puck_indices))
    return puck_information

def get_puck_centers(present, not_present, number_of_pucks=9):
    indices = np.vstack((present, not_present))
    
    if len(indices) > number_of_pucks:
        n_clusters = number_of_pucks
    else:
        n_clusters = len(indices)
    
    puck_kmeans = KMeans(n_clusters=n_clusters)
    puck_kmeans.fit(indices)
        
    puck_centers = puck_kmeans.cluster_centers_
    return puck_centers

def main():
    parser = optparse.OptionParser()
    
    parser.add_option('-i', '--imagename', default='image.jpg', type=str, help='path to the image to search for objects in')
    parser.add_option('-s', '--detect_pins', action="store_true", dest="pins", default=False,  help='Detect pins')
    parser.add_option('-c', '--puck_window', default=119, type=int, help='puck detection window size')
    parser.add_option('-n', '--pin_window', default=27, type=int, help='pin detection window size')
    parser.add_option('-o', '--puck_step_size', default=7, type=int, help='puck detection stride')
    parser.add_option('-d', '--pin_step_size', default=2, type=int, help='pin detection stride')
    parser.add_option('-U', '--puck_classifier', default='puck_detector.pickle', type=str, help='pickled puck classifier')
    parser.add_option('-I', '--pin_classifier', default='pin_detector.pickle', type=str, help='pickled pin classifier')
    parser.add_option('-r', '--results_string', default='_detect5', type=str, help='results string')
    parser.add_option('-D', '--display', default=False, dest='display', action="store_true", help='Show results of detection')
    parser.add_option('-m', '--images_info', default='images_info.pickle', type=str, help='Dictionary with training images information')
    
    options, args = parser.parse_args()
    
    print options.imagename
    
    database = pickle.load(open(options.images_info))
    if database[options.imagename]['scale'] == 'dewar':
        pass
    else:
        print options.imagename, 'scale is not that of dewar'
        sys.exit(0)
    image = unsharp_and_local_equalize(img_as_float(imread(options.imagename, as_grey=True)))
    
    rgb_image = imread(options.imagename)
    
    puck_segment_size = options.puck_window 
        
    puck_classifier = pickle.load(open(options.puck_classifier))
    
    present, not_present = detect_object(image, puck_segment_size, options.puck_step_size, puck_classifier)
    
    puck_centers = get_puck_centers(present, not_present)
      
    present_puck_indices, present_puck_locations, weights = determine_centers_corresponding_to_present_objects(puck_centers, present, not_present, distance=10)
  
    puck_centers_shifted = puck_centers + puck_segment_size/2
    
    fig = plt.figure(figsize=(10, 8.2))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(rgb_image)
    
    lid_centers, lids_kmeans = get_lid_centers(puck_centers_shifted)
    
    average_lid_distance = get_average_lid_distance(lid_centers)
    
    lid_indices_reverse_index, lid_indices  = get_lid_indices(lid_centers)
    
    puck_predictions = lids_kmeans.predict(puck_centers_shifted)
    
    puck_information = get_puck_information(puck_centers_shifted, puck_predictions, lid_indices_reverse_index, puck_segment_size)
    
    present_puck_order_vs_location = {}
    for n in range(3):
        for k, coordinate in enumerate(puck_centers_shifted[puck_predictions==n]):
            r, c = coordinate
            puck_local = puck_information[n][0][k] 
            puck_global = puck_information[n][1][puck_local]['puck_global']
            puck_coordinate = puck_information[n][1][puck_local]['center']
            present_puck_order_vs_location[puck_coordinate] = puck_global
            
    lid_predictions = lids_kmeans.predict(lid_centers)
  

    for l, coordinate in enumerate(lid_centers):
        r, c = coordinate
        circ = plt.Circle((c, r), radius=average_lid_distance/2.3, edgecolor='m', lw=2, facecolor='none')
        ax.annotate(lid_indices_reverse_index[l], (c, r-average_lid_distance/2.28), color='m', fontsize=10)
        ax.add_patch(circ)
    
    for n, lc in enumerate(lid_centers):
        for k, coordinate in enumerate(puck_centers_shifted[puck_predictions==n]):
            r, c = coordinate
            if  np.any(np.abs(np.sum(present_puck_locations - (coordinate - puck_segment_size/2), axis=1)) < 0.001):
                color = 'green'
            else:
                color = 'red'
            circ = plt.Circle((c, r), radius=average_lid_distance/7, edgecolor=color, lw=2, facecolor='none')
            
            puck_local = puck_information[n][0][k] 
            puck_global = puck_information[n][1][puck_local]['puck_global']
            ax.annotate('%s(%d)' % (puck_local.replace('puck',''), puck_global), (c-puck_segment_size/9, r-average_lid_distance/7), color='w', fontsize=8)
            ax.add_patch(circ)
          
    if options.pins == False:
        if options.display:
            plt.show()
        ax.set_title(options.imagename)
        ax.set_axis_off()
        plt.savefig(options.imagename.replace('.jpg', '%s.jpg' % options.results_string), dpi='figure')
        sys.exit(0)
    
    crop_size = average_lid_distance/3.
    pin_classifier = pickle.load(open(options.pin_classifier))
    
    all_present_pins = np.array([])
    all_missing_pins = np.array([])
    pin_indices = np.array([])
    pin_labels = np.array([])
    pin_presence_flags =[]
    pin_absence_flags = []
    
    kmeans_pin_indices = KMeans(n_clusters=16)
    print 'average_lid_distance', average_lid_distance
    for k, location in enumerate(present_puck_locations):
        location = tuple(map(int, location))

        puck_label = present_puck_order_vs_location[location]

        subwindow = get_subwindow(image, location, options.puck_window)
        pin_present, pin_missing = detect_object(subwindow, options.pin_window, options.pin_step_size, pin_classifier)
        
        pin_present += np.array(location)
        pin_missing += np.array(location)
        
        all_present_pins = np.vstack([all_present_pins, pin_present]) if all_present_pins.size else pin_present
        all_missing_pins = np.vstack([all_missing_pins, pin_missing]) if all_missing_pins.size else pin_missing

        pin_locations = np.vstack((pin_present, pin_missing))
        
        pin_kmeans = KMeans(n_clusters=16)
        pin_kmeans.fit(pin_locations)
        pin_centers = pin_kmeans.cluster_centers_
        
        scale, maximum, priemer, minimum, center = get_scale_and_center_from_pin_positions(pin_centers)
        
        if options.display:
            print 'scale', scale
            print 'priemer', priemer
            print 'maximum', maximum
            print 'minimum', minimum
            print 'center', center
            print 'lid distance vs. pin distance', average_lid_distance/scale
        
        distances = np.apply_along_axis(norm, 1, pin_locations - center)
        pin_locations = pin_locations[distances/priemer < 1.4]
        distances = np.apply_along_axis(norm, 1, pin_locations - center)
        pin_locations = pin_locations[0.4 < distances/priemer]
        
        pin_kmeans.fit(pin_locations)
        pin_centers = pin_kmeans.cluster_centers_
        
        scale, maximum, priemer, minimum, center = get_scale_and_center_from_pin_positions(pin_centers)
        
        rotated_and_scaled = rotate_and_scale_and_shift_ideal_pin_coordinates(puck_angles[puck_label], scale, center)
        
        kmeans_pin_indices.cluster_centers_ = rotated_and_scaled
                
        predicted_pin_indices = kmeans_pin_indices.predict(pin_centers) + 1
        
        if options.display:
            print 'scale', scale
            print 'priemer', priemer
            print 'maximum', maximum
            print 'minimum', minimum
            print 'center', center
            print 'lid distance vs. pin distance', average_lid_distance/scale
            print
        
        pin_labels = np.vstack([pin_labels, predicted_pin_indices]) if pin_indices.size else predicted_pin_indices
        pin_indices = np.vstack([pin_indices, pin_centers]) if pin_indices.size else pin_centers
        #pin_indices = np.vstack([pin_indices, kmeans_pin_indices.cluster_centers_]) if pin_indices.size else pin_centers
        
        present_pin_indices, present_pin_locations, weights = determine_centers_corresponding_to_present_objects(pin_centers, pin_present, pin_missing, distance=4)
        ppf = np.zeros(len(pin_centers))
        ppf[present_pin_indices] = weights
        ppf = list(ppf)
        pin_presence_flags += ppf
        
        absent_pin_indices, absent_pin_locations, weights = determine_centers_corresponding_to_present_objects(pin_centers, pin_missing, pin_present, distance=4)
        paf = np.zeros(len(pin_centers))
        paf[absent_pin_indices] = weights
        paf = list(paf)
        pin_absence_flags += paf
        
    pin_presence_flags = np.array(pin_presence_flags)
    pin_absence_flags = np.array(pin_absence_flags)
    pin_info = np.hstack((pin_indices, pin_labels.reshape((pin_labels.size, 1)), pin_presence_flags.reshape((pin_presence_flags.size,1)), pin_absence_flags.reshape((pin_absence_flags.size,1)) ))
    
    for k, coordinate in enumerate(all_present_pins):
        edgecolor = 'green'
        r, c = coordinate
        r += options.pin_window/2. - options.pin_window/20
        c += options.pin_window/2. - options.pin_window/20
        rect = plt.Rectangle((c, r), options.pin_window/10, options.pin_window/10, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
    
    for k, coordinate in enumerate(all_missing_pins):
        edgecolor = 'red'
        r, c = coordinate
        r += options.pin_window/2. - options.pin_window/20  # - options.pin_window/4 + 2
        c += options.pin_window/2. - options.pin_window/20 #- options.pin_window/4 + 2
        rect = plt.Rectangle((c, r), options.pin_window/10, options.pin_window/10, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
    
    for k, coordinate_label in enumerate(pin_info):
        r, c, label, present, absent = coordinate_label
        r += options.pin_window/4.
        c += options.pin_window/4.
        if present >= 1 and absent == 0:
            edgecolor = 'green'
        elif present == 0 and absent >= 1:
            edgecolor = 'red'
        elif present > absent:
            edgecolor = 'green'
        elif present < absent:
            edgecolor = 'red'
        else:
            edgecolor = 'orange'
        rect = plt.Rectangle((c, r), options.pin_window/2, options.pin_window/2, edgecolor=edgecolor, facecolor='none')
        ax.annotate('%d' % label, (c, r))
        ax.add_patch(rect)
        
    ax.set_title(options.imagename)
    ax.set_axis_off()
    plt.savefig(options.imagename.replace('.jpg', '%s.jpg' % options.results_string), dpi='figure')
    if options.display:
        plt.show()
        
if __name__ == '__main__':
    main()
    
