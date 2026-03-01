from StringArtEngine import StringArtEngine
import time
from PIL import Image
import numpy as np
import os

### Main string art generation script (this should do everything you need) ###
# default path for saving project is results/recent
# save a project by copying it into results/library/your_project

image_name = 'example.png' # image must be in images folder

# Specify your desired setup then simply run the file
settings = {
    # image preprocessing
    "use_importance": True, # draw an importance mask to highlight detail in the string art
    "background_removal": False, # additional setup parameters are in StringArtEngine __init__
    "background_color": 100, # 100, if we remove background, replace with this shade (255=white)
    "darkening": 0.8, # 0.8, darkening image can improve accuracy
                        
    # physical correlation and layout
    "thread_type": "nylon", # 0.1mm nylon monofilament
    "board_diameter_mm": 480, # adjust this to your desired string art size
    "num_nails": 200, # 180-240 total number of nails
    "pattern": 'square', # circle or square
    "save_template": True, # this will save a high res template for your setup
}
# after generation, you can use number_reader.py for tracking your progress (it will autosave your current index)

################################################################################################
# Main funtion calls

t0 = time.time()
Engine = StringArtEngine(
                         background_removal= settings["background_removal"],
                         background_color= settings["background_color"],
                         darkening= settings["darkening"],
                         thread_type= settings["thread_type"], 
                         board_diameter_mm= settings["board_diameter_mm"],
                         num_nails= settings["num_nails"], 
                         pattern= settings["pattern"], 
                         )
use_importance = settings["use_importance"]
save_template = settings["save_template"]
original_image = Image.open('images/' + image_name)  
target = Engine.preprocess(original_image) # Step 1, prepare image for string art generation
t1 = time.time()

if use_importance:
    try:
        importance = np.load("results/recent/importance.npy")
    except:  
        importance = np.ones_like(target, dtype=np.float32)    
    print("Waiting for importance mask...")
    importance = Engine.draw_importance_mask(target, importance)
else:
    importance = np.ones_like(target, dtype=np.float32)


t2 = time.time()
print("Generating...")
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
np.savetxt("results/recent/sequence.txt", np.array([sequence]), fmt="%d", delimiter=",") # basic txt
Engine.build_sequence_pdf(sequence, out_path="results/recent/sequence.pdf") # pdf version
# Save template
if save_template:
    nail_coords = Engine.create_nail_positions(Engine.num_nails, 3000, Engine.pattern)
    Engine.make_template(nail_coords)
else:
    try:
        os.remove("results/recent/nail_template.png")
    except:
        pass
t5 = time.time()

tPreprocess = round(t1-t0,2)
tGenerate = round(t3-t2,2)
tRender = round(t4-t3,2)
tSave = round(t5-t4,2)
tTotal = round(tPreprocess + tGenerate + tRender + tSave,2)
print(f'Prepare: {tPreprocess}s, Generate: {tGenerate}s, Render: {tRender}s, Save: {tSave}s, TOTAL: {tTotal}s')
print(f'Number of lines: {len(sequence)}')
print('FINISHED - saved to results/recent')