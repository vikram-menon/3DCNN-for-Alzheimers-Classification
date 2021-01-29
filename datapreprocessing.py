import os
import cv2
import numpy as np
from PIL import Image

src = 'sourcefilelocation'
vids = ['vid1.mp4', 'vid2.mp4', 'vid3.mp4', ....]

count = 0
i = 0

for f in vids:
    name = vids[count]
    cap = cv2.VideoCapture(src + name)
    
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dist = framecount-26     
    sframe = dist/2
    cap.set(cv2.CAP_PROP_POS_FRAMES, sframe)
    
    framect = 0;

    # 24-36 credit to: moshel
    while(cap.isOpened()):
        ret, frame = cap.read()
        img = frame
        if framect == 26:
            ret = False
        if ret == False:
            break  
        th = cv2.inRange(img, (7, 13, 104), (98, 143, 255))                                                                                                                
        points = np.where(th>0)                                                                                                                                              
        p2 = zip(points[0], points[1])                                                                                                                                       
        p2 = [p for p in p2]                                                                                                                                                 
        rect = cv2.boundingRect(np.float32(p2))                                                                                                                              
        cv2.rectangle(img, (rect[1], rect[0]), (rect[1]+rect[3], rect[0]+rect[2]), 0)

        height = (rect[0]+rect[2]) - rect[0]
        width = (rect[1]+rect[3]) - rect[1]
        y = rect[0]
        x = rect[1]
        roi = img[y:y+height, x:x+width]
        
        # 45-57 credit to: https://gist.github.com/jdhao
        desired_size = 48
        im = roi
        old_size = im.shape[:2]
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        im = cv2.resize(im, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        
        cv2.imwrite('savefilelocation'+str(i)+'.jpg', new_im)
        
        image = Image.open('savefilelocation'+str(i)+'.jpg')
        image_data = image.load()
        height,width = image.size
        for loop1 in range(height):
            for loop2 in range(width):
                r,g,b = image_data[loop1,loop2]
                if(r > -1 and r < 256 and g > -1 and g < 160 and b > -1 and b<100):
                    image_data[loop1,loop2] = 0,0,0
        image.save('savefilelocation'+str(i)+'.jpg')
        
        i += 1
        framect += 1;
    count += 1
cap.release()
cv2.destroyAllWindows()
