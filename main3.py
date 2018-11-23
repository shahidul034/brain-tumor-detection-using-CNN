import h5py
import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)

files={}
array = {}
cnt=0
for i in range(2299,3065):
    filepath=r"C:\Users\User\Desktop\data\brainTumorDataPublic_2299-3064/" + str(i) + '.mat'
    f=h5py.File(filepath)
    x=f['cjdata']
    img=np.array(x['image'])
    #type(img)
    #img
    #cv2.imwrite('test'+str(i)+'.jpg',img)

    array['img']=np.array(x['image'])
    array['label']=int(np.array(x['label'])[0][0])
    array['tumormask'] = np.array(x['tumorMask'])
    if array['label'] == 1 :
        cv2.imwrite(r"C:\Users\User\Desktop\data\L1/" + 'test' + str(i) + '.jpg', img)
    if array['label'] == 2 :
        cv2.imwrite(r"C:\Users\User\Desktop\data\L2/" + 'test' + str(i) + '.jpg', img)
    if array['label'] == 3 :
        cv2.imwrite(r"C:\Users\User\Desktop\data\L3/" + 'test' + str(i) + '.jpg', img)
    files[i]=array
