from PIL import Image
import numpy as np
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size, Image.BICUBIC)
    return img

def postprocess_output(img_y, img_cb, img_cr):
    img_out_y = Image.fromarray(np.uint8((img_y * 255.0).clip(0, 255)[0]), mode='L')
    return Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC)
        ]
    ).convert("RGB")

