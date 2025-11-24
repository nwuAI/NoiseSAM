import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import random
import os
import argparse

def add_poisson_noise(image, lam=200, scale_factor=1):
    img_array = np.array(image).astype(np.float32) / 255.0
    scaled_img = img_array * scale_factor
    noise_level = lam / 255.0
    signal_with_poisson = np.random.poisson(scaled_img)
    noise = (signal_with_poisson - scaled_img) / scale_factor
    noisy_img = img_array + noise * noise_level
    noisy_img_uint8 = np.clip(noisy_img, 0, 1) * 255
    return Image.fromarray(noisy_img_uint8.astype(np.uint8))

def reduce_contrast(image, factor=0.1):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.0 - factor)

def add_gaussian_noise(image, variance=0.3):
    img_array = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, variance, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 1)
    return Image.fromarray((noisy_img * 255).astype(np.uint8))

def add_salt_and_pepper_noise(image, density_range=(0.2, 0.2)):
    img_array = np.array(image)
    density = random.uniform(*density_range)
    num_salt = np.ceil(density * img_array.size * 0.6)
    num_pepper = np.ceil(density * img_array.size * 0.6)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape[:2]]
    img_array[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape[:2]]
    img_array[coords[0], coords[1]] = 0
    return Image.fromarray(img_array)

def rotate_image(image, angle_range=(17, 17)):
    angle = random.uniform(*angle_range)
    return image.rotate(angle, resample=Image.BICUBIC, expand=False)

def add_gaussian_blur(image, kernel_size=7):
    return image.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2))

def preprocess_and_perturb_image(image):
    image = image.resize((216, 216), resample=Image.BICUBIC)
    perturbations = [
        add_salt_and_pepper_noise,
        add_gaussian_blur,
    ]
    num_perturbations = random.randint(2, 2)
    selected_perturbations = random.sample(perturbations, num_perturbations)
    for perturbation in selected_perturbations:
        image = perturbation(image)
    return image

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            try:
                image = Image.open(input_path).convert('RGB')
                processed_image = preprocess_and_perturb_image(image)
                output_path = os.path.join(output_folder, filename)
                processed_image.save(output_path)
                print(f"已处理: {filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量处理图像并添加噪声')
    parser.add_argument('--input', required=True, help='输入文件夹路径')
    parser.add_argument('--output', required=True, help='输出文件夹路径')
    args = parser.parse_args()
    process_images_in_folder(args.input, args.output)    