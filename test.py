from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load processor & model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image
image = Image.open("C:/Users/2003b/OneDrive/Pictures/Camera Roll 1/WIN_20220324_09_28_16_Pro.jpg")

# ----- 1. Normal caption (no prompt) -----
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
print("Caption:", processor.decode(out[0], skip_special_tokens=True))

# ----- 2. Prompted caption -----
prompt = "what is the type of image , give me answer in one word?"
inputs = processor(images=image, text=prompt, return_tensors="pt")
out = model.generate(**inputs)
print("Prompted Answer:", processor.decode(out[0], skip_special_tokens=True))
