from modeltrained import Model

import cv2
import numpy as np
import dlib
from math import hypot
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import time
ana = Model()

sec=30
files={"video":'./Data/Room8/Math/2023-03-24.mp4'}
st = time.time()
for key, value in files.items():
            cap = cv2.VideoCapture(value)
            ana = Model()
            Key=key
            # Capture every frame and send to detector
            i=0
            F=1
            coloms=['Frames','Focused','Not Focused']
            Data=[]
            Data2=[]
            while True:
                _, frame = cap.read()
                bm,tot,foc,nfoc = ana.detect_face(frame,i,Key)
                i=i+1
                print("Key",i,Key)
                key = cv2.waitKey(1)
                if tot>0:
                    res=[]
                    res2=[]
                    res.append('F'+str(F))
                    res.append((foc/tot)*100.0)
                    res.append((nfoc/tot)*100.0)
                    Data.append(res)
                    res2.append('F'+str(F))
                    res2.append((foc))
                    res2.append((nfoc))
                    Data2.append(res2)
                    F=F+1
                et = time.time()
                elapsed_time = et - st
                if elapsed_time>=sec:
                    break

            # Exit if 'q' is pressed
                if key == ord('q'):
                    break
                if i==500:
                    break
            
            df = pd.DataFrame(Data,
                        columns=coloms)
            df.plot(x='Frames', kind='bar', stacked=True,
                title='Results Percentage')
            plt.savefig('res.png')

            df2 = pd.DataFrame(Data2,
                        columns=coloms)
            df2.plot(x='Frames', kind='bar', stacked=True,
                title='Results Percentage')
            plt.savefig('res2.png')
            # Release the memory
            cap.release()
            cv2.destroyAllWindows()