# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import glob
import numpy as np
from feature_extraction import extract_features



classes = {
    "1":"፩",
    "2":"፪",
    "3" :"፫",
    "4" :"፬",
    "5" :"፭",
    "6" :"፮",
    "7" :"፯",
    "8" :"፰",
    "9" :"፱",
    "10" :"፲",
    "11" :"፳",
    "12" :"፴",
    "13" :"፵",
    "14" :"፶",
    "15" :"፷",
    "16" :"፸",
    "17" :"፹",
    "18" :"፺",
    "19" :"፻",
    "20" :"፼",
}




if __name__== '__main__':
    ## Here goes the generation of a dataset
    dir_names  = classes.keys()
    target = []
    data = []
    for name in dir_names:
        paths = glob.glob('paths/{}/*'.format(name))
        
        for path in paths:
            points = np.load(path)
            features = extract_features(points)
            
            data.append(features)
            target.append(int(name))
            
    
    with open('datasets/geez-numeral-target.npz', 'w+') as f:
        np.save(f, np.array(target))
    
    with open('datasets/geez-numeral-data.npz', 'w+') as f:
        np.save(f, np.array(data))
        


