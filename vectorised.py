import cv2, os, numpy
from PIL import Image
import svgwrite
import subprocess
import xml.etree.ElementTree as ET

def raster_to_vector(input_file, output_file):
    command = f'potrace {input_file} -s -o {output_file}'
    subprocess.run(command, shell=True)
def extract_numbers(text):
    return [float(num) for num in text.split() if num.replace('.', '', 1).isdigit()]

path = "Content/flowers/"
transfer = "Content/transfer/"
savepath = "Content/vectors/"
ims = []
for name in os.listdir(path):
    im = Image.open(path+name)
    im.save(transfer+name[:-4]+".bmp")
for name in os.listdir(transfer):
    print(raster_to_vector(transfer+name, savepath+name[:-4]+".svg"))
    text = open(savepath+name[:-4]+".svg",mode="r").read()
    numbers = []
    print(numbers)
