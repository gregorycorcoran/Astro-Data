from glob import glob
import os
from multiprocessing import Pool
with open('AstrometryAPIKey.txt') as f:
    key=f.readlines()[0]
objects=glob('./CoAdds/*')
def Runfunc(object):
    obj_name=object.split('/')[-1]
    #print(obj_name)
    images=glob(object+'/*.fits')
    for image in images:
        #print(image.lstrip('./'))
        filt_name=image.split('/')[-1]
        #print(filt_name)
        new_loc=f'WCS_Solved/{obj_name}/{obj_name}_{filt_name}'
        for attempt in range(10):
            try:
                if os.path.isfile(new_loc)==False:
                    os.system(f'python astrometry.net/astrometry/net/client/client.py -u {image.lstrip('./')} -k {key} --newfits {new_loc}')
            except TimeoutError:
                continue
            break
with Pool(7) as p:
    p.map(Runfunc,objects)