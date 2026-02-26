from StringArtEngine import StringArtEngine
import time
from PIL import Image
import numpy as np
import os

### Main string art generation script (this should do everything you need) ###

image_name = 'example.png' # in images folder
 
Engine = StringArtEngine(
                         # image preprocessing
                         background_removal= False, # additional setup parameters are in StringArtEngine __init__
                         background_color= 100, # 100, if we remove background, replace with this shade (255=white)
                         darkening= 0.8, # 0.8, darkening image can improve accuracy
                                              
                         # physical correlation
                         thread_type="nylon", # 0.1mm nylon monofilament
                         board_diameter_mm=480, # adjust this to your desired string art size
                         num_nails=200, # 180-240 total number of nails
                         pattern='circle', # circle, square
                         )
use_importance = False # draw an importance mask to highlight detail in the string art

######################################################################################################
# Useful utilities (uncomment whichever one you need)

# # Use this to make nail templates (select pattern and number of nails above)
# # (physical size depends on how you slice it - use rasterbater.net with 10mm margin and 5mm overlap)
# nail_coords = Engine.create_nail_positions(Engine.num_nails, 3000, Engine.pattern) # need higher resolution for nail coords
# Engine.make_template(nail_coords)
# quit()

# # Use this to make new line profiles (then put in assets)
# profiles = Engine.precompute_line_profiles(Engine.nail_coords)
# np.save(f'line_profiles_{Engine.resolution}.npy',profiles)
# quit()

# # Use this to generate a preview from a sequence
# sequence = np.loadtxt('results/recent/sequence.txt', delimiter=',', dtype=int)
# previews, _ = Engine.render_all_previews(sequence)
# for name, img in previews.items():
#     img.convert("RGB").save(f'results/renders/high_quality_render_{name}.jpeg')
# quit()

# # Use this to convert image into png
# img = Image.open("results/recent/preview_outside.jpeg")
# img.convert("RGBA").save("pngimage.png")
# quit()

# # Use this for circular cropping
# img = Image.open('images/test/image.png')
# circular_img = Engine.circular_crop_rgba(img.convert("RGBA"))
# circular_img.save("circular.png")
# quit()

#####################################################################################
# Main function calls 

t0 = time.time()
original_image = Image.open('images/' + image_name)  
target = Engine.preprocess(original_image) # Step 1, prepare image for string art generation
t1 = time.time()

if use_importance:
    try:
        importance = np.load("results/recent/importance.npy") # try to load the last used mask
    except:  
        importance = np.ones_like(target, dtype=np.float32)    
    importance = Engine.draw_importance_mask(target, importance)
else:
    importance = np.ones_like(target, dtype=np.float32)

t2 = time.time()
sequence = Engine.generate_sequence_numba(target, use_importance, importance) # Step 2, generate sequence from image
t3 = time.time()
previews, plain_render = Engine.render_all_previews(sequence) # Step 3, turn sequence into accurate renders
t4 = time.time()

##########################################################################################
# Saving results

# Save original image
square_img = Image.fromarray(Engine.largest_square(np.array(original_image)))
square_img.convert('RGB').save("results/recent/original_image.jpeg")
if Engine.pattern == "circle":
    circle_img = Engine.circular_crop_rgba(square_img)
    circle_img.save("results/recent/original_image_circle.png")
else:
    try:
        os.remove("results/recent/original_image_circle.png")
    except:
        pass
# Save target
target_img = Image.fromarray((target * 255).astype(np.uint8))
target_img.convert('L').save('results/recent/processed_target.jpeg', format="jpeg")
# Save importance mask
if use_importance:
    np.save("results/recent/importance.npy", importance)
else:
    try:
        os.remove("results/recent/importance.npy")  
        os.remove("results/recent/importance.jpeg")
    except:
        pass
# Save previews
for name, img in previews.items():
    path = f"results/recent/preview_{name}.jpeg"
    img.convert('RGB').save(path, format="jpeg")
plain_render.save('results/recent/final_render.png')
# Save sequence
np.savetxt("results/recent/sequence.txt", np.array([sequence]), fmt="%d", delimiter=",")
t5 = time.time()

tPreprocess = round(t1-t0,2)
tGenerate = round(t3-t2,2)
tRender = round(t4-t3,2)
tSave = round(t5-t4,2)
tTotal = round(tPreprocess + tGenerate + tRender + tSave,2)
print(f'Preprocess: {tPreprocess}s, Generate: {tGenerate}s, Render: {tRender}s, Save: {tSave}s, TOTAL: {tTotal}s')
print(f'Number of lines: {len(sequence)}')
print('FINISHED - saved to results/recent')