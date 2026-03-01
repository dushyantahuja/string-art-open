import numpy as np
from PIL import Image
from skimage import color, exposure
from skimage.transform import resize
from skimage.draw import line_aa, disk
from importlib import resources
import random
import numba
from pathlib import Path
from StringArtUtils import StringArtUtils
    
@numba.njit
def generate_sequence_numba_func(
    target,
    use_importance,
    importance,   
    candidate_nails,
    line_rr,
    line_cc,
    line_val,
    line_start,
    line_end,
    line_map,
    line_strength,
    max_lines
):
    residual = np.ones_like(target, dtype=np.float32) - target

    current = 0
    path = np.empty(max_lines, dtype=np.int32)
    path_len = 0

    while True:
        best = -1
        best_imp = 0.0

        for k in range(candidate_nails.shape[1]):
            i = candidate_nails[current, k]
            if i < 0: # code for invalid index (too close or path clashes)
                break

            line_id = line_map[current, i] # use this id to index the line profiles array
            if line_id < 0:
                continue

            start = line_start[line_id]
            end = line_end[line_id]

            dot_r_val = 0.0
            sum_val_sq = 0.0

            if use_importance:
                for p in range(start, end):
                    rr = line_rr[p]
                    cc = line_cc[p]
                    v = line_val[p]
                    r = residual[rr, cc]
                    w  = importance[rr, cc]

                    dot_r_val += w * r * v
                    sum_val_sq += w * v * v
            else:
                for p in range(start, end):
                    rr = line_rr[p]
                    cc = line_cc[p]
                    v  = line_val[p]
                    r  = residual[rr, cc]

                    dot_r_val += r * v
                    sum_val_sq += v * v
                    
            imp = 2.0 * line_strength * dot_r_val - (line_strength * line_strength) * sum_val_sq

            if imp > best_imp:
                best_imp = imp
                best = i

        if best == -1 or path_len >= max_lines:
            break

        line_id = line_map[current, best]
        start = line_start[line_id]
        end = line_end[line_id]

        for p in range(start, end):
            rr = line_rr[p]
            cc = line_cc[p]
            residual[rr, cc] -= line_strength * line_val[p]

        path[path_len] = best
        path_len += 1
        current = best

    return path[:path_len]

class StringArtEngine(StringArtUtils):
    """
    Main functions and algorithm for string art generation.
    Use the StringArtUtils superclass for additional functions.
    """

    def __init__(self, 
                 background_removal = False, 
                 background_color = 100, # 80-140, changes background shade
                 darkening = 0.8, # 0.7-0.9, improves results quite a lot
                 clip_limit = 0.02, # 0.02-0.04, CLAHE image preprocessing, doesn't need changing
                 thread_type = 'nylon', # 0.1mm nylon monofilament
                 max_lines = 5500, # will break before this when improvement stops
                 board_diameter_mm = 480, # can change for your needs
                 num_nails = 200, # total
                 min_distance = 15, # consecutive indices in sequence must be this far apart to avoid short paths
                 resolution = 500, # 500 - don't change, it won't make string art any better unless canvas is huge
                 pattern = 'circle', # circle, square
                 ):
        super().__init__()
        self.background_removal = background_removal
        self.background_color = background_color
        self.clip_limit = clip_limit
        self.darkening = darkening
        self.thread_type = thread_type
        self.max_lines = max_lines     
        self.board_diameter_mm = board_diameter_mm      
        self.num_nails = num_nails
        self.min_distance = min_distance
        self.resolution = resolution  
        self.pattern = pattern
        
        # Thread correlations (correlated at 500 resolution and with 480mm board diameter)
        if self.thread_type == "nylon": # 0.1mm nylon monofilament
            self.kThread = 0.075
            self.line_strength = self.kThread * self.resolution/500 * 480/board_diameter_mm
        elif self.thread_type == "poly": # 40/2 thin polyester thread
            self.kThread = 0.3
            self.line_strength = self.kThread * self.resolution/500 * 480/board_diameter_mm
        else:
            raise ValueError("Invalid thread type")
        
        if self.resolution not in [500]:
            raise ValueError("Increasing resolution won't improve results (unless your canvas is huge)")
        
        # Calculate nail coordinates and valid paths and load in line profiles and templates
        self.nail_coords = self.create_nail_positions(self.num_nails, self.resolution, self.pattern)
        self.candidate_nails = self.get_candidate_nails(self.nail_coords, self.num_nails, self.min_distance, self.pattern)
        self.line_profiles = self.load_or_make_line_profiles(self.resolution, self.pattern, self.nail_coords)
        self.templates = self.load_templates()
    
    def load_or_make_line_profiles(self, resolution, pattern, nail_coords, num_nails):
        name = f"line_profiles_{pattern}_{resolution}_{num_nails}.npy"
        assets_path = Path(resources.files("assets"))
        file_path = assets_path / name
        
        if not file_path.exists():
            print("Line profiles not found — computing and saving new ones")
            profiles = self.precompute_line_profiles(nail_coords)
            assets_path.mkdir(parents=True, exist_ok=True)
            np.save(file_path, profiles)
        else:
            print("Loaded line profiles")

        return np.load(file_path, allow_pickle=True).item()

    def load_templates(self):
        print('Loaded preview templates')
        def load(name):
            with resources.files(
                "assets"
            ).joinpath(name).open("rb") as f:
                return Image.open(f).convert("RGBA")

        return {
            "easel": load("template_easel.png"),
        }
    
    def largest_square(self, image):
        """
        Input and output are np arrays
        """
        h, w = image.shape[:2]
        if h <= w:
            s = (w - h) // 2
            return image[:, s : s + h]
        else:
            s = (h - w) // 2
            return image[s : s + w, :]
    
    def compute_string_length_km(
        self,
        sequence,
    ):
        mm_per_pixel = self.board_diameter_mm / self.resolution
        total_length_mm = 0.0

        prev_idx = sequence[0]
        y0, x0 = self.nail_coords[prev_idx]
        for idx in sequence[1:]:
            y1, x1 = self.nail_coords[idx]
            dx = x1 - x0
            dy = y1 - y0
            dist_px = np.sqrt(dx**2 + dy**2)
            total_length_mm += dist_px * mm_per_pixel
            x0, y0 = x1, y1
        total_length_km = round(total_length_mm * 1e-6, 2)
        return total_length_km

    # --------------------------------------------------
    # Core algorithm
    # --------------------------------------------------

    def preprocess(self, image_rgb: Image.Image):
        image_rgb = image_rgb.convert("RGB")
        
        # 1. Background removal (must come first to work effectively)
        if self.background_removal:
            from rembg import remove 
            rgba = remove(image_rgb, bgcolor=(self.background_color,)*3)
            rgb = rgba.convert("RGB")
        else:
            rgb = image_rgb

        # 2. Grayscale
        target = color.rgb2gray(rgb).astype(np.float32)

        # 3. Contrast limited adaptive histogram equalisation (insanely good)
            # kernel size can be made bigger to reduce leakage into background, default is 1/8 of image size
            # clip limit is super important, larger value increases contrast, but too much contrast
            # can ruin the final look by creating artificial light and dark regions
        target = exposure.equalize_adapthist(target, clip_limit=self.clip_limit)

        # 4. Crop and resize
        target = self.largest_square(target)
        target = resize(target, (self.resolution, self.resolution), anti_aliasing=True)

        # 5. Robust normalisation
        p_low, p_high = np.percentile(target, (2, 98))
        target = np.clip(target, p_low, p_high)
        target = (target - p_low) / (p_high - p_low + 1e-6)
        
        # 6. Darkening (really useful) - adds more lines to final piece
        target *= self.darkening
                 
        return target # np array, 0-1

    def generate_sequence_numba(self, target, use_importance = False, importance = 0):
        """
        Numba compiled string art generator (~1s).
        """
        if use_importance == False:
            importance = np.ones_like(target, dtype=np.float32) # safe deafult

        seq = generate_sequence_numba_func(
            target,
            use_importance,
            importance, 
            self.candidate_nails,
            self.line_profiles["rr"],
            self.line_profiles["cc"],
            self.line_profiles["val"],
            self.line_profiles["start"],
            self.line_profiles["end"],
            self.line_profiles["map"],
            self.line_strength,
            self.max_lines
        )
        return seq

    # --------------------------------------------------
    # Preview generation
    # --------------------------------------------------

    def place_on_template(self, template, coords, art):
        """
        Place art on template
        """
        (cx, cy), diameter = coords

        # Resize while preserving alpha
        art_resized = art.resize(
            (diameter, diameter),
            resample=Image.LANCZOS
        ).convert("RGBA")

        x = int(cx - diameter / 2)
        y = int(cy - diameter / 2)

        composite = template.copy().convert("RGBA")
        composite.alpha_composite(art_resized, (x, y))

        return composite
   
    def apply_lighting(self, render, brightness=1.0, tint=(1.0, 1.0, 1.0)):
        arr = np.array(render, dtype=np.float32) / 255.0
        rgb = arr[..., :3]
        alpha = arr[..., 3:]
        rgb = rgb * brightness
        rgb = rgb * tint
        rgb = np.clip(rgb, 0, 1)
        out = np.concatenate([rgb, alpha], axis=-1)
        return Image.fromarray((out * 255).astype(np.uint8), mode="RGBA")
    
    def render_all_previews(self, sequence):
        # Define template coordinates (center (x,y from top left), diameter)
        template_easel_coords = ([1260, 1024], 850)

        LIGHTING = {
            "easel": {"brightness": 1.2,  "tint": (1.0, 1.0, 1.0)},
        }

        plain_render = self.render_from_sequence(sequence)

        # Helper to apply lighting based on preset name
        def render_with_lighting(name):
            preset = LIGHTING.get(name, {})
            return self.apply_lighting(
                plain_render,
                brightness=preset.get("brightness", 1.0),
                tint=preset.get("tint", (1.0, 1.0, 1.0))
            )

        # Can easily add more custom previews with different lighting (instructions in ReadMe)
        previews = {
            "easel": self.place_on_template(self.templates["easel"], template_easel_coords, render_with_lighting("easel")),
        }

        return previews, plain_render
         
    def render_from_sequence(
        self,
        sequence,
        render_resolution=1000, # 1000-1500                
        supersample = 3,  # 2 (good) or 3 (best)
        jitter_mm=2.5,    # 2-3, removes aliasing
        ):
        """     
        Render a high-quality string art preview from a nail sequence.
        Uses line_aa and a 0-1 float canvas which enables control of darkness by line_strength.
        Generates nail_coords directly at supersampled resolution.
        """
        
        W = render_resolution * supersample
        mm_2_px = W / self.board_diameter_mm
        
        # --- Generate nail coordinates directly at supersampled resolution ---
        nail_coords = self.create_nail_positions(self.num_nails, W, self.pattern)
        
        # Determine correct line strength for the thread type, resolution, and board diameter
        line_strength = self.kThread * W / 500 * 480 / self.board_diameter_mm

        # Background canvas
        canvas = np.ones((W, W), dtype=np.float32)

        # --- Draw nail discs ---
        nail_radius_mm = 1.5  
        nail_radius_px = nail_radius_mm * mm_2_px

        for y, x in nail_coords:
            cy = int(round(y))
            cx = int(round(x))
            r = int(round(nail_radius_px))
            if r <= 0:
                continue
            rr, cc = disk((cy, cx), r, shape=canvas.shape)
            canvas[rr, cc] = 0

        # --- Jitter in pixels ---
        max_jitter_px = jitter_mm * mm_2_px
        def jitter(p):
            return (
                p[0] + random.uniform(-max_jitter_px, max_jitter_px),
                p[1] + random.uniform(-max_jitter_px, max_jitter_px)
            )

        # --- Draw lines ---
        prev_idx = sequence[0]
        for idx in sequence[1:]:
            p0_j = jitter(nail_coords[prev_idx])
            p1_j = jitter(nail_coords[idx])

            y0, x0 = int(round(p0_j[0])), int(round(p0_j[1]))
            y1, x1 = int(round(p1_j[0])), int(round(p1_j[1]))

            rr, cc, val = line_aa(y0, x0, y1, x1)

            valid = (rr >= 0) & (rr < W) & (cc >= 0) & (cc < W)
            rr, cc, val = rr[valid], cc[valid], val[valid]

            canvas[rr, cc] = np.clip(canvas[rr, cc] - line_strength * val, 0, 1)
            prev_idx = idx

        # --- Convert canvas to PIL image ---
        img = Image.fromarray((canvas * 255).astype(np.uint8)).convert("RGBA")

        # --- Downsample for anti-aliasing ---
        if supersample > 1:
            img = img.resize((render_resolution, render_resolution), resample=Image.LANCZOS)

        # --- Optional circular crop ---
        if self.pattern == "circle":
            img = self.circular_crop_rgba(img)

        return img