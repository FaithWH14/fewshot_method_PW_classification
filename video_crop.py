import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

import json
from pandas import json_normalize
import pandas as pd
import glob


mount_dir = "/mnt/c/Users/faithwh14/Desktop/PW1/video" #video zip files path
list_videos = os.listdir(mount_dir) #list all inside this video folders
file_name = "groundtruth.json" #select the person_groundtruth.json
videos = [] #collect all mp4 inside the path

# function to calculate the area of an image or pixel
def area(crops):
    area_ = [crop.shape[0] * crop.shape[1] for crop in crops]
    return np.mean(area_), area_

for idx, i in enumerate(list_videos):
    if not os.path.isdir(os.path.join(mount_dir, i)): #exclude zip files
        continue
    elif not os.path.exists(os.path.join(mount_dir, i, file_name)): #exclude some folders without person_groundtruth.json
        continue
    with open(os.path.join(mount_dir, i, file_name)) as f: #otherwise, load the person_groundtruth.json
        groundtruth = next(iter(json.load(f))) #unlist the list with single element

    
    try:
        frames = groundtruth["frames"]

        list_groundtruthINFO = []
        for frame in frames:
            groundtruth_info = next(iter(frame["groundtruth"])) #unlist the single element list
            groundtruth_info["num"] = frame["num"]
            groundtruth_info["timestamp"] = frame["timestamp"]
            list_groundtruthINFO.append(groundtruth_info)

        df = json_normalize(list_groundtruthINFO)
        video = glob.glob(os.path.join(mount_dir, i, "*.mp4")) #extract the mp4 files
        videos.append(video)

        cat = cv2.VideoCapture(video[0])
        
        selected_frames = df["num"].tolist()
        top_left = list(zip(df.rawTLx.tolist(), df.rawTLy.tolist()))
        btm_right = list(zip(df.rawBRx.tolist(), df.rawBRy.tolist()))

        dict1 = {}
        for idx2, i2 in enumerate(selected_frames):
            dict1[i2] = [top_left[idx2], btm_right[idx2]]

        ret_ = True
        count = 0
        hi = np.zeros([100])
        gg = []
        wp = []
        while ret_:
            ret_ = False
            ret, frame_ = cat.read()
            ret_ = ret
            if (ret_ is not True) | (count == 10000):
                break

            if count in selected_frames:
                gg.append(frame_)
                wp.append(dict1[count])
            count += 1

        gg2 = np.stack(gg, axis = 0)

        images = []
        crops = []
        for i3, (j,k) in zip(gg, wp):
            ii3 = i3
            #cv2.imshow("test", i)
            if k[1] - j[1] > 35:
                crops.append(ii3[j[1]:k[1], j[0]:k[0]])
            img1 = cv2.rectangle(i3, j, k, (43, 234, 23), 1)
            cv2.imshow("ret_", img1)
            
            cv2.waitKey(30)
            images.append(cv2.cvtColor(i3, cv2.COLOR_BGR2RGB))

        #cap.release()
        cv2.destroyAllWindows()
        imageio.mimsave("/mnt/c/Users/faithwh14/Desktop/PW1/people_{}.gif".format(idx), images)
        
        mean_crops, area_crops = area(crops)
        crops1 = [i5 for idx5, i5 in enumerate(crops) if area_crops[idx5] >= mean_crops]

        crops_path = "/mnt/c/Users/faithwh14/Desktop/PW1/crops{}".format(idx)
        if not os.path.exists(crops_path):
            os.makedirs(crops_path)

        for idx4, i4 in enumerate(crops1):
            cv2.imwrite(os.path.join(crops_path, "{}_crops.jpg".format(idx4)), i4)

    except Exception:
        pass

