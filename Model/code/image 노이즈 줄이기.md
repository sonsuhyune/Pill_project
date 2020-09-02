## image 노이즈 줄이기

[참고 블로그](https://eehoeskrap.tistory.com/125)



- 노이즈가 많은 이미지의 엣지를 부각시키기 위한 이미지 전처리



**Bilateral Filter**

: 선형으로 처리되지 않고, 엣지와 노이즈를 줄여주어 부드러운 영상이 만들어지게

: bilateralFilter(src,dst,d,sigmaColor,sigmaSpace);

 



```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('135_B_2_W_L_1_1.jpg')
blur = cv.bilateralFilter(img,9,50,50)
sobelx = cv.Sobel(blur,cv.CV_8U,1,0,ksize=3)
sobely = cv.Sobel(blur,cv.CV_8U,0,1,ksize=3)
bitwise_or = cv.bitwise_or(sobelx,sobely)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(bitwise_or),plt.title('Blurred+')
plt.xticks([]), plt.yticks([])
plt.show()
```

