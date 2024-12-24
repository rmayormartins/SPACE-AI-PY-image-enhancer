import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import gradio as gr
import numpy as np
import tempfile
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(scale):
    model = RealESRGAN(device, scale=scale)
    weights_path = f'weights/RealESRGAN_x{scale}.pth'
    try:
        model.load_weights(weights_path, download=True)
        print(f"Weights for scale {scale} loaded successfully.")
    except Exception as e:
        print(f"Error loading weights for scale {scale}: {e}")
        model.load_weights(weights_path, download=False)
    return model

model2 = load_model(2)
model4 = load_model(4)
model8 = load_model(8)

def enhance_image(image, scale):
    try:
        print(f"Enhancing image with scale {scale}...")
        start_time = time.time()
        image_np = np.array(image.convert('RGB'))
        print(f"Image converted to numpy array: shape {image_np.shape}, dtype {image_np.dtype}")
        
        if scale == '2x':
            result = model2.predict(image_np)
        elif scale == '4x':
            result = model4.predict(image_np)
        else:
            result = model8.predict(image_np)
            
        enhanced_image = Image.fromarray(np.uint8(result))
        print(f"Image enhanced in {time.time() - start_time:.2f} seconds")
        return enhanced_image
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return image

def muda_dpi(input_image, dpi):
    dpi_tuple = (dpi, dpi)
    image = Image.fromarray(input_image.astype('uint8'), 'RGB')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    image.save(temp_file, format='PNG', dpi=dpi_tuple)
    temp_file.close()
    return Image.open(temp_file.name)

def resize_image(input_image, width, height):
    image = Image.fromarray(input_image.astype('uint8'), 'RGB')
    resized_image = image.resize((width, height))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    resized_image.save(temp_file, format='PNG')
    temp_file.close()
    return Image.open(temp_file.name)

def process_image(input_image, enhance, scale, adjust_dpi, dpi, resize, width, height):
    original_image = Image.fromarray(input_image.astype('uint8'), 'RGB')
    
    if enhance:
        original_image = enhance_image(original_image, scale)
    
    if adjust_dpi:
        original_image = muda_dpi(np.array(original_image), dpi)
        
    if resize:
        original_image = resize_image(np.array(original_image), width, height)
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    original_image.save(temp_file.name)
    return original_image, temp_file.name

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="Upload"),
        gr.Checkbox(label="Enhance Image (ESRGAN)"),
        gr.Radio(['2x', '4x', '8x'], type="value", value='2x', label='Resolution model'),
        gr.Checkbox(label="Adjust DPI"),
        gr.Number(label="DPI", value=300),
        gr.Checkbox(label="Resize"),
        gr.Number(label="Width", value=512),
        gr.Number(label="Height", value=512)
    ],
    outputs=[
        gr.Image(label="Final Image"),
        gr.File(label="Download Final Image")
    ],
    title="Image Enhancer",
    description="Upload an image (.jpg, .png), enhance using AI, adjust DPI, resize and download the final result.",
    examples=[
        ["gatuno.JPG"]
    ]
)
# ....
iface.launch(debug=True)
