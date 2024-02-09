from PIL import Image
import os

filepath = "results/flowers_c/"
frames = []

for frame_name in range(345):
    frame = Image.open(filepath+f"result{frame_name*10}.jpg")
    frames.append(frame)

frames[0].save(
    'process.gif',
    save_all=True,
    append_images=frames[1:],
    optimize=True,
    duration=200,
    loop=0
)