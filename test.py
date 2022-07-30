import numpy as np
from PIL import Image

import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F

fake_img = Image.fromarray((np.random.rand(4, 5, 3) * 255.).astype(np.uint8))

print(type(fake_img))

transform = T.ToTensor()

tensor = transform(fake_img)

print(tensor.shape)
# [3, 4, 5]

print(tensor.dtype)
# paddle.float32
