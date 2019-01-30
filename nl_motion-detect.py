#!/usr/bin/env python
# coding: utf-8

# ### Wykrywanie obszarów ruchu

# In[44]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray


#path = './pobrane/'

im1 = io.imread('ziarna1.jpg')
im2 = io.imread('ziarna2.jpg')


plt.imshow(im1)
plt.show()

plt.imshow(im2)
plt.show()


# In[45]:


import cv2

diff = cv2.absdiff(im1, im2)

gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap="gray")
plt.title('Grayscale')
plt.show()

ret, diff2 = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)

plt.imshow(diff)
plt.title('Original RGB')
plt.show()

plt.imshow(diff2, cmap="gray")
plt.title('Binary')
plt.show()


# In[46]:


import numpy as np

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(diff2, cv2.MORPH_CLOSE, kernel)

cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
# cnts = imutils.grab_contours(cnts)

minArea = 50
margin = 3

for c in cnts:
    if cv2.contourArea(c) > minArea:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(im2, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 0, 255), 2)
        print('Status: object detected')
        print(x, y, w, h)

        plt.imshow(im2, interpolation='lanczos')
        plt.title('Contours')
        plt.show()
        
        print(im2.shape)
        


# ### Wykrywanie obszarów ruchu
# #### Analiza poklatkowa na przykładzie strumienia wideo
# ####    

# In[47]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('man-walk.mp4')

counter = 1
fnum = [100, 101, 102]
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == False:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if counter in fnum:
        plt.imshow(gray, cmap='gray')
        plt.show()

    counter += 1
        
cap.release()


# In[48]:


# from imutils.video import VideoStream
# import imutils
import numpy as np

def motionDetect(frame, num):

    bg = 255*np.ones_like(frame)
    
    diff = cv2.absdiff(frame, bg)

    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    ret, diff2 = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(diff2, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    minArea = 50
    margin = 3

    for c in cnts:
        if cv2.contourArea(c) > minArea:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 0, 255), 2)
            print('Frame no.: ' + str(num))
            print('Status: object detected')
            print('Motion area: ', x, y, w, h, end="\n\n")

    plt.figure()
    plt.imshow(frame, interpolation='lanczos')
    plt.title('Contours')
    plt.show()


# In[49]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('man-walk.mp4')

counter = 1
fnum = list(range(100,103))

prevFrame = None

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == False:
        break

    if counter in fnum:
        motionDetect(frame, counter)

    counter += 1
        
cap.release()


# ### Metoda uśredniania tła :: wyznaczanie modelu tła
# 
# <ol>
# <li style="margin: 3px">Obliczamy średni obraz tła</li>
# <li style="margin: 3px">Obliczamy odchylenie standardowe tła (przybliżenie: średnie różnice, FFAAD)</li>
#     + FFAAD (ang. Frame-to-Frame Average Absolute Difference)
# <li style="margin: 3px">Ustalamy progi względem wielokrotności FFAAD:</li>
#     + HighThreshold
#     + LowThreshold
# <li style="margin: 3px">Odejmujemy poszczególne klatki obrazu i porównujemy z modelem tła</li>    
# <li style="margin: 3px">Piksele nie pasujące do modelu tła uznajemy za obszar obiektów w ruchu</li>    
# </ol>

# In[61]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('IMG-3031.MOV')

counter = 1
# fnum = [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
fnum = [range(1,50)]
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == False:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if counter in fnum:
        plt.imshow(gray, cmap='gray')
        plt.show()
        bg = frame

    counter += 1
        
cap.release()


# In[72]:


# from imutils.video import VideoStream
# import imutils
import numpy as np

def motionDetect(frame, num, bg):

#     bg = 255*np.ones_like(frame)
    
    diff = cv2.absdiff(frame, bg)

    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    ret, diff2 = cv2.threshold(gray,90,255,cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(diff2, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    minArea = 150
    margin = 3

    for c in cnts:
        if cv2.contourArea(c) > minArea:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 0, 255), 2)
            print('Frame no.: ' + str(num))
            print('Status: object detected')
            print('Motion area: ', x, y, w, h, end="\n\n")

    plt.figure()
    plt.imshow(frame, interpolation='lanczos')
    plt.title('Contours')
    plt.show()


# In[73]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('IMG-3031.MOV')

counter = 1
fnum = list(range(55,70))

prevFrame = None

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == False:
        break

    if counter in fnum:
        motionDetect(frame, counter, bg)

    counter += 1
        
cap.release()


# In[ ]:




