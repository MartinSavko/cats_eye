#!/usr/bin/env python

'''draw the clicks'''

import pygame
#from pygame.locals import *
import sys

from skimage.transform import rescale
from skimage.io import imread, imsave
from skimage.util import crop

import numpy as np
import random
import seaborn as sns

xkcd_colors_that_i_like = ["pale purple", "coral", "moss green", "windows blue", "amber", "greyish", "faded green", "dusty purple", "crimson", "custard", "orangeish", "dusk blue", "ugly purple", "carmine", "faded blue", "dark aquamarine", "cool grey", "faded blue"]

from cats_eyes import detect_pucks, get_average_lid_distance, get_lid_centers, get_puck_centers, determine_centers_corresponding_to_present_objects

from prepare_examples import unsharp_and_local_equalize

def get_closest_click(click, clicks):
    nclicks= np.array(clicks)
    nclick = np.array(click)
    a = np.apply_along_axis(np.linalg.norm, 1, nclicks - nclick)
    return clicks[np.argmin(a)]

def draw_current_points(screen, puck_present_clicks, puck_not_present_clicks, pin_present_clicks, pin_not_present_clicks, scale=None):
    for k, clicks in enumerate([puck_present_clicks, puck_not_present_clicks, pin_present_clicks, pin_not_present_clicks]):
        for pos in clicks:
            if scale != None:
                pos = int(round(pos[0]*scale)), int(round(pos[1]*scale))
            pygame.draw.circle(screen, tuple(int(sns.xkcd_rgb[xkcd_colors_that_i_like[k]][1:][i:i+2], 16) for i in (0, 2 ,4)), pos, 2)
            
def draw_rectangle_around_detected_pucks(screen, puck_centers, present, segment_size, average_lid_distance, scale=None):
    not_present = []
    for i in puck_centers:
        if i not in present:
            not_present.append(i)
    
    if scale != None:
        segment_size *= scale
        average_lid_distance *= scale
        
    colors = ['moss green', 'carmine']
    for k, category in enumerate([present, not_present]):
        for pos in category:
            y, x = pos
            if scale != None:
                x, y = int(round(x*scale)), int(round(y*scale))
            x += segment_size/2
            y += segment_size/2
            x -= average_lid_distance/2
            y -= average_lid_distance/2
            pygame.draw.rect(screen, tuple(int(sns.xkcd_rgb[colors[k]][1:][i:i+2], 16) for i in (0, 2 ,4)), (x, y, average_lid_distance, average_lid_distance), 2)

def get_clicks(record):
    clicks = []
    for k, item in enumerate(['puck_present', 'puck_not_present', 'pin_present', 'pin_not_present', 'pin_negative']):
        try:
            clicks.append(list(set(record[item])))
        except:
            clicks.append([])
   
    return clicks

def main():
    import optparse
    import pickle
    
    parser = optparse.OptionParser()
    parser.add_option('-d', '--database', default='images_scales.pickle', help='pickled dictionary with image filenames and corresponding information about scale and keypoints (click coordinates)')
    parser.add_option('-f', '--filename', default='./example_image.jpg', type=str, help='image to examine')
    parser.add_option('-t', '--target', default='pin', type=str, help='what object are we looking for: pin, puck or lid?')
    parser.add_option('-r', '--rescale', default=None, type=float, help='rescale the image and click coordinates')
    parser.add_option('-c', '--classifier', default='puck_classifier.pickle')
    parser.add_option('-w', '--puck_window', default=119, type=int, help='puck detection window size')
    options, args = parser.parse_args()
    
    print options.filename
    
    database = pickle.load(open(options.database))
    
    if type(database[options.filename]) == dict:
        if database[options.filename]['scale'] == 'dewar':
            pass
        else:
            print 'Image not that of the dewar'
            sys.exit(0)
    else:
        s = database[options.filename]
        print 's', s
        database[options.filename] = {}
        database[options.filename]['scale'] = s
        f = open(options.database, 'w')
        pickle.dump(database, f)
        f.close()
        print 'Image not that of the dewar'
        sys.exit(0)
    
    #if '2017-05-27' in options.filename or '2017-06-02' in options.filename:
        #print 'skipping', options.filename
        #sys.exit(0)
        
    classifier = pickle.load(open(options.classifier))
    
    segment_size = options.puck_window
    
    image = unsharp_and_local_equalize(imread(options.filename, as_grey=True))
    
    present, not_present = detect_pucks(image, segment_size, 7, classifier) 
    
    puck_centers = get_puck_centers(present, not_present)
    
    present_puck_indices, present_pucks, weights = determine_centers_corresponding_to_present_objects(puck_centers, present, distance=10)
    
    puck_centers_shifted = puck_centers + segment_size/2
    
    lid_centers, lids_kmeans = get_lid_centers(puck_centers_shifted)
    average_lid_distance = get_average_lid_distance(lid_centers)
    print 'average_lid_distance', average_lid_distance
    x, y = image.shape[::-1]
    
    print 'scale factor', options.rescale
    if options.rescale != None:
        image = imread(options.filename)
        imager = rescale(image, options.rescale)
        y, x, c = imager.shape
        imsave('/tmp/imagr.jpg', imager)
    
    puck_present_clicks, puck_not_present_clicks, pin_present_clicks, pin_not_present_clicks, pin_negative_clicks = get_clicks(database[options.filename])
    
    present = None
    print 'present_pucks'
    print present_pucks
    
    crop_size = average_lid_distance/3.
    for puck in present_pucks:
        #options.rescale = round(8*116/average_lid_distance, 1)
        
        pygame.init()
        
        image = imread(options.filename) #, as_grey=True)
        
        y, x = puck + segment_size/2 #+ segment_size/2

        before_y = int(y - crop_size/2)
        after_y = int(y + crop_size/2 + 1)
        before_x = int(x - crop_size/2)
        after_x = int(x + crop_size/2 + 1)
        
        puck_image = image[before_y: after_y, before_x: after_x, ::]
        scale = round(400./average_lid_distance, 1)
        puck_image = rescale(puck_image, scale)
        
        imsave('/tmp/puck.jpg', puck_image)
        
        #if options.rescale == None:
            #image = pygame.image.load(options.filename)
        #else:
        
        screen = pygame.display.set_mode(puck_image.shape[:2][::-1], 0, 32)
        pygame.display.set_caption('determine the key points')
        image = pygame.image.load('/tmp/puck.jpg')
        
        screen.blit(image, (0,0 ))
        pygame.display.update()
        move_on = True
        while move_on:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    move_on = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    x, y = pygame.mouse.get_pos()
                    print 'user clicked at', x, y
                    sox = int(before_x + x/scale)
                    soy = int(before_y + y/scale)
                    print 'this correspond to coordinate', sox, soy
                    if present == True:
                        pin_present_clicks.append((sox, soy))
                    elif missing == True:
                        pin_not_present_clicks.append((sox, soy))
                    elif negative == True:
                        pin_negative_clicks.append((sox, soy))

                elif event.type == pygame.KEYDOWN and event.unicode == 'p':
                    present = True
                    missing = False
                    negative = False
                    print 'detecting present s'
                    
                elif event.type == pygame.KEYDOWN and event.unicode == 'm':
                    present = False
                    missing = True
                    negative = False
                    print 'detecting empty pin positions'
                
                elif event.type == pygame.KEYDOWN and event.unicode == 'n':
                    present = False
                    missing = False
                    negative = True
                    print 'detecting empty pin positions'
                
                elif event.type == pygame.KEYDOWN and event.unicode == 's':
                    present = False
                    print 'saving clicks to database'
                    database[options.filename]['pin_present'] = list(set(pin_present_clicks))
                    database[options.filename]['pin_not_present'] = list(set(pin_not_present_clicks))
                    database[options.filename]['pin_negative'] = list(set(pin_negative_clicks))
                    f = open(options.database, 'w')
                    pickle.dump(database, f)
                    f.close()
                #elif event.type == pygame.KEYDOWN and event.unicode in ['d', 'l', 'u']:
                    #if event.unicode == 'd':
    f = open(options.database, 'w')
    pickle.dump(database, f)
    f.close()
    sys.exit()
                
    #if event.type == pygame.MOUSEBUTTONUP:
        #x, y = pygame.mouse.get_pos()
        #print 'user clicked at', x, y
        #click_to_remove = get_closest_click((x, y), puck_present_clicks)
        #print len(puck_present_clicks)
        #a = puck_present_clicks.index(click_to_remove)
        #print 'removing click', click_to_remove
        #print puck_present_clicks.pop(a)
        #database[options.filename]['puck_present'] = puck_present_clicks
                    
if __name__ == '__main__':
    main()
