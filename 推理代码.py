#依赖
from modelscope.pipelines import pipeline

#模型地址
model_old_dir="./model_old"
model_new_dir="./model_new"

#模型加载
ocr_detection_old = pipeline(Tasks.ocr_detection, model=model_old_dir)
ocr_detection_new = pipeline(Tasks.ocr_detection, model=model_new_dir)

#模型推理
result_old = ocr_detection_old('imgs/test.png')
result_new = ocr_detection_new('imgs/test.png')

#可视化
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("imgs/test.png")
draw = ImageDraw.Draw(image)

polygons = result_old["polygons"]
for polygon in polygons:
    coords = [(int(x), int(y)) for x, y in polygon.reshape((-1, 2))]
    draw.polygon(coords, outline="red")

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off") 
plt.show()
