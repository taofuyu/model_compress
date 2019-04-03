import time
import os

def log(filename, content):
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    with open(filename,'a') as f:
        f.write(time_stamp + ': '+ content+'\n')

def compute_mAP(model, test_imgs_path):
