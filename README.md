# 简介
- 微调阿里开源的文字检测模型，利用合合识别返回的OCR结果作为初始训练数据，对模型进行优化训练，使其更加适应1万张图片的具体场景，提高文字区域检测的精度，优化边界框质量，减少漏检和误检。
- 后期逐步开放10万张，100万张和1000万张图片微调以后的模型



# 依赖
from modelscope.pipelines import pipeline

# 模型地址
model_old_dir="./model_old"
model_new_dir="./model_new"

# 模型加载
ocr_detection_old = pipeline(Tasks.ocr_detection, model=model_old_dir)
ocr_detection_new = pipeline(Tasks.ocr_detection, model=model_new_dir)

# 模型推理
result_old = ocr_detection_old('imgs/test.png')
result_new = ocr_detection_new('imgs/test.png')

# 可视化
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
