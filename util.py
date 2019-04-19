import time
import os
from PIL import Image, ImageDraw
import numpy as np
from yolo import YOLO

def log(filename, content):
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    with open(filename,'a') as f:
        f.write(time_stamp + ': '+ content+'\n')

def detect_and_save(yolo_model, raw_img_path, out_path):
    img_files = os.listdir(raw_img_path)
    img_files.sort()
    for img in img_files:
        raw_img = Image.open(raw_img_path+img)
        #use YOLO member function to perform detection process
        boxes, scores, classes = yolo_model.detect_image(raw_img)
        thickness = (raw_img.size[0] + raw_img.size[1]) // 300
        #to cal mAP, save results to .txt file. 
        if len(classes)>0:
                detect_r_txt = open(out_path+img.split('.')[0]+'.txt','w+')
                for i, c in reversed(list(enumerate(classes))):
                        to_write = ''
                        predicted_class = yolo_model.class_names[c]
                        box = boxes[i]
                        score = scores[i]
                        #pre draw
                        label = '{} {:.2f}'.format(predicted_class, score)
                        draw = ImageDraw.Draw(raw_img)
                        label_size = draw.textsize(label)
                        #get coor values
                        top, left, bottom, right = box
                        top = max(0, np.floor(top + 0.5).astype('int32'))
                        left = max(0, np.floor(left + 0.5).astype('int32'))
                        bottom = min(raw_img.size[1], np.floor(bottom + 0.5).astype('int32'))
                        right = min(raw_img.size[0], np.floor(right + 0.5).astype('int32'))
                        #write prob and coor to file to cal mAP
                        to_write = str(predicted_class)+' '+str(score)+' '+str(left)+' '+str(top)+' '+str(right)+' '+str(bottom)
                        detect_r_txt.write(to_write+'\n')
                        #draw box and label on img, then save. If need show, use img.show()
                        if top - label_size[1] >= 0:
                                text_origin = np.array([left, top - label_size[1]])
                        else:
                                text_origin = np.array([left, top + 1])
                        # My kingdom for a good redistributable image drawing library.
                        for j in range(thickness):
                                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=yolo_model.colors[c])

                        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=yolo_model.colors[c])
                        draw.text(tuple(text_origin), label, fill=(0, 0, 0))
                        del draw
                        raw_img.save(out_path + img)
                detect_r_txt.close()

        
