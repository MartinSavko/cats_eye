#!/usr/bin/env python

'''draw the clicks'''

import pygame
import sys

from skimage.transform import rescale
from skimage.io import imread, imsave
from skimage.util import crop

import numpy as np
import random
import seaborn as sns

from cats_eyes import detect_pucks, get_average_lid_distance, get_lid_centers

from prepare_examples import unsharp_and_local_equalize

xkcd_colors_that_i_like = ["pale purple", "coral", "moss green", "windows blue", "amber", "greyish", "faded green", "dusty purple", "crimson", "custard", "orangeish", "dusk blue", "ugly purple", "carmine", "faded blue", "dark aquamarine", "cool grey", "faded blue"]
click_category_codes = ['1', '2', '3', '4', '5', '6', 'a']
click_names = ['puck_present', 'puck_not_present', 'puck_negative', 'pin_present', 'pin_not_present', 'pin_negative']

def get_closest_click(click, clicks):
    nclicks= np.array(clicks)
    nclick = np.array(click)
    a = np.apply_along_axis(np.linalg.norm, 1, nclicks - nclick)
    return clicks[np.argmin(a)]

def draw_current_points(screen, click_categories, category_to_plot='a', scale=None):
    colors = ["pale purple", "coral", "faded blue", "windows blue", "crimson", "cool grey"]
    
    for k, clicks in enumerate(click_categories):
        color_rgb = tuple(int(sns.xkcd_rgb[colors[k]][1:][i:i+2], 16) for i in (0, 2 ,4))
        print click_names[k], clicks
        if len(clicks)>0 and category_to_plot=='a':
            for pos in clicks:
                if scale != None:
                    pos = int(round(pos[0]*scale)), int(round(pos[1]*scale))
                pygame.draw.circle(screen, color_rgb, pos, 2)
        elif len(clicks)>0 and str(k+1) == category_to_plot:
            for pos in clicks:
                if scale != None:
                    pos = int(round(pos[0]*scale)), int(round(pos[1]*scale))
                pygame.draw.circle(screen, color_rgb, pos, 2)
        else:
            pass
        
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
    for k, item in enumerate(click_names):
        try:
            clicks.append(list(set(record[item])))
        except:
            clicks.append([])
   
    return clicks


def get_negative_clicks(database, imagename, nnegative=300, boundary=1, nmindistance=7, nmaxdistance=65, target='pin'):
    present = list(set(database[imagename]['%s_present' % target]))
    not_present = list(set(database[imagename]['%s_not_present' % target]))
    positive_clicks = present + not_present
    
    negative = []
    
    while len(negative) < nnegative:
        x_index = random.randint(0+boundary, 704-boundary)
        y_index = random.randint(0+boundary, 576-boundary)
        cc = get_closest_click((x_index, y_index), positive_clicks)
        if np.linalg.norm(np.array([x_index, y_index]) - np.array(cc)) > nmindistance and np.linalg.norm(np.array([x_index, y_index]) - np.array(cc)) < nmaxdistance:
            negative.append((x_index, y_index))
    
    return present, not_present, negative

def main():
    import optparse
    import pickle
    
    parser = optparse.OptionParser()
    parser.add_option('-d', '--database', default='images_scales.pickle', help='pickled dictionary with image filenames and corresponding information about scale and keypoints (click coordinates)')
    parser.add_option('-f', '--filename', default='./experimental_images/2017-05-27/12_05_19/2017-05-27_12:05:19_pan-17.4724_tilt-87.0000_zoom5800_merged.jpg', type=str, help='image to examine')
    options, args = parser.parse_args()

    #if '2017-05-27' in options.filename or '2017-06-02' in options.filename:
        #print 'skipping', options.filename
        #sys.exit(0)
        
    print 'filename', options.filename
    print 'record'
    database = pickle.load(open(options.database))
    record = database[options.filename]
    print record
    
    if type(database[options.filename]) == dict:
        pass
    else:
        database[options.filename] = {'scale': record}
        f = open(options.database, 'w')
        pickle.dump(database, f)
        f.close()
    
    puck_present_clicks, puck_not_present_clicks, puck_negative, pin_present_clicks, pin_not_present_clicks, pin_negative = get_clicks(pickle.load(open(options.database))[options.filename])

    #cpresent, cmissing, cnegative = get_negative_clicks(pickle.load(open(options.database)), options.filename, nnegative=1500, boundary=1, nmindistance=7, nmaxdistance=45, target='pin')
                                                        
    pygame.init()
    
    image = imread(options.filename)
    
    screen = pygame.display.set_mode(image.shape[:2][::-1], 0, 32)
    
    image = pygame.image.load(options.filename)
    
    screen.blit(image, (0, 0))
                
    pygame.display.update()
    
    move_on = True
    category_to_plot = 'a'
    delete = None
    ajout = None
    while move_on:
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                move_on = False
            
            elif event.type == pygame.KEYDOWN and event.unicode == 'r':
                print 'refreshing clicked points'
                clicks = get_clicks(pickle.load(open(options.database))[options.filename])
                screen.blit(image, (0, 0))
                #clicks[-1] = cnegative
                draw_current_points(screen, clicks, category_to_plot=category_to_plot)
                pygame.display.update()
            
            elif event.type == pygame.KEYDOWN and event.unicode in click_category_codes:
                clicks = get_clicks(pickle.load(open(options.database))[options.filename])
                category_to_plot = event.unicode
                print 'selecting category', category_to_plot
                
            elif event.type == pygame.MOUSEBUTTONUP and delete == True:
                database = pickle.load(open(options.database))
                clicks = get_clicks(database[options.filename])
                if category_to_plot != 'a':
                    relevant_category_of_clicks = clicks[click_category_codes.index(category_to_plot)]
                    clicks_name = click_names[click_category_codes.index(category_to_plot)]
                else:
                    print 'can not remove points if display set to "a". Please select specific category in 1, 2, 3, 4, 5 and 6'
                    continue
                if len(relevant_category_of_clicks) == 0:
                    print clicks_name, 'is empty'
                    continue
                x, y = pygame.mouse.get_pos()
                print 'user clicked at', x, y
                click_to_remove = get_closest_click((x, y), relevant_category_of_clicks)
                print len(relevant_category_of_clicks)
                a = relevant_category_of_clicks.index(click_to_remove)
                print 'removing click', click_to_remove
                print relevant_category_of_clicks.pop(a)
                
                database[options.filename][clicks_name] = relevant_category_of_clicks
                f = open(options.database, 'w')
                pickle.dump(database, f)
                f.close()
                
            elif event.type == pygame.MOUSEBUTTONUP and ajout == True:
                database = pickle.load(open(options.database))
                clicks = get_clicks(database[options.filename])
                if category_to_plot != 'a':
                    relevant_category_of_clicks = clicks[click_category_codes.index(category_to_plot)]
                    clicks_name = click_names[click_category_codes.index(category_to_plot)]
                else:
                    print 'can not remove points if display set to "a". Please select specific category in 1, 2, 3, 4, 5 and 6'
                x, y = pygame.mouse.get_pos()
                print 'adding click at', x, y
                relevant_category_of_clicks.append((x, y))
                database[options.filename][clicks_name] = relevant_category_of_clicks
                f = open(options.database, 'w')
                pickle.dump(database, f)
                f.close()
            
            elif event.type == pygame.KEYDOWN and event.unicode in ['+', '-']:
                print 'Changing the list of clicks'
                if event.unicode == '-':
                    print 'click on the point you want to remove'
                    delete = True
                    ajout = False
                elif event.unicode == '+':
                    print 'click on the keypoint to add'
                    delete = False
                    ajout = True
                    
if __name__ == '__main__':
    main()