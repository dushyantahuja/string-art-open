Author: Will Taylor
Learn more: stringboard.co.uk

An Open Source Python Based String Art Generation Repository
* fast: 1-2s generation with numba compiler
* accurate: previews correlated to real string art 
* optimal: importance weighting, robust image preprocessing, optimal greedy algorithm

SETUP AND DEPENDENCIES
git clone https://github.com/StringBoardUK/string-art-open.git
cd string-art-open
pip install -r requirements.txt
(or just use VS code quick create virtual environment)

Note: rembg is quite heavy and can cause issues when importing,
if you don't need background removal, comment it out in StringArtEngine imports

===========================================================================================
CONTENTS

main.py: main script for converting an image into string art sequence and rendering a result

StringArtEngine has the core logic for string art generation
    * image preprocessing pipeline
    * numba compiled sequence generator, compatible with importance weighting
    * result renderer - accurate and well parameterised 

StringArtUtils has useful functions:
    * making nail templates
    * generating line profiles (anti aliased line coordinates)
    * UI for importance weighting

number_reader: simple UI for reading sequence (helpful for actually stringing)

assets
    - line_porfiles_500.npy: pre computed anti-aliased line profiles (rows,cols,vals), 150Mb
    (will be created on the first generation, then reused)
    - nail_template.png: full size nail template
    - nail_template_sliced.pdf: sliced for 50cm board diameter
    - template_board.jpeg: template for rendering to show realistic result on real board (200 nails 50cm diameter)

========================================================================================================
Technical details and usage:

* This string art generator has been developed by correlating physical results to model parameters, 
the best thread type: 0.1mm nylon monofilament. With this thread you should aim for 3.5-4.5k lines for 50cm board

* Important Additional setup parameters live in the StringArtEngine class itself (e.g. board diameter, number of nails)

* Circular only (can easily refactor to do other shapes, just need to generate nail coordinates
and the new line profiles data)

* Resolution of input image to generator does not need to be above 500x500px. String art
cannot achieve higher than 200-300px resolution, so input doesn't need to be more

* Lines are drawn 1px thick (+anti aliasing), and line strength is calculated as follows:
line_strength = kNylon * resolution/500 * 480/board_diameter_mm
kNylon - 0.1mm nylon thread constant, and then adjusted for resolution and board diameter (nominal 1)
e.g. if you wanted to use a different thread change kNylon to a new weighting parameter

* Once sequence is generated, the code will render the final string art into a high resolution (1000x1000px) png image. 
This can then be pasted onto a template image with known centre and diameter coordinates
(I've provided template_wall, but you can use your own). Lighting can be simulated by adjusting brightness and tint

* To add a new template image for rendering
- in StringArtEngine load_templates_cached, load in new template image
- in StringArtEngine render_all_previews, add the new template coords, lighting setup and add to previews dictionary

* Use the importance UI tool to highlight details in the string art. Can adjust the strength of this in StringArtUtils

* Do not change folder structure (it will break stuff!)