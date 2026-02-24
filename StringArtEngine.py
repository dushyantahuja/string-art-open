import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage import color, exposure
from skimage.transform import resize
from skimage.draw import line_aa
from importlib import resources
import random
from functools import lru_cache
import numba
import sys
from pathlib import Path
from rembg import remove # can comment this out if you don't need background removal

# this is just so we don't need a duplicated assets folder in base directory (when running main.py)
API_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(API_ROOT))

@lru_cache(maxsize=1)
def load_line_profiles_cached(resolution: int):
    name = f"line_profiles_{resolution}.npy"
    with resources.files("assets").joinpath(name).open("rb") as f:
        print("Loaded line profiles")
        return np.load(f, allow_pickle=True).item()

@lru_cache(maxsize=1) 
def load_templates_cached():
    print('Loaded preview templates')
    def load(name):
        with resources.files(
            "assets"
        ).joinpath(name).open("rb") as f:
            return Image.open(f).convert("RGBA")

    return {
        "wall": load("template_wall.jpeg"),
    }
    
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
            if i < 0:
                break

            line_id = line_map[current, i]
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

class StringArtEngine:
    """
    Main functions and algorithm for string art generation.
    One instance per request.
    """

    def __init__(self, 
                 background_removal = False,
                 background_color = 100, # 80-140   
                 darkening = 0.8, # 0.7-0.9
                 clip_limit = 0.02, # 0.02-0.04
                 thread_type = 'nylon',
                 max_lines = 5500,     
                 board_diameter_mm = 480,
                 num_nails = 200,
                 min_distance = 15,
                 resolution = 500,
                 ):
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
        
        # Thread correlations (correlated at 500 resolution and with 480mm board diameter)
        self.kNylon = 0.075 # 0.1mm nylon monofilament
        self.kPoly = 0.3 # 40/2 thin polyester thread

        if self.thread_type == "nylon": 
            self.line_strength = self.kNylon * self.resolution/500 * 480/board_diameter_mm
        elif self.thread_type == "poly": 
            self.line_strength = self.kPoly * self.resolution/500 * 480/board_diameter_mm
        else:
            raise ValueError("Invalid thread type")
        
        if self.resolution not in [500]:
            raise ValueError('Incorrect resolution')
        
        # Precompute valid candidate nails for each nail index (using min distance constraint)
        self.candidate_nails = [
            [
                j for j in range(self.num_nails)
                if min(abs(i - j), self.num_nails - abs(i - j)) > self.min_distance
            ]
            for i in range(self.num_nails)]
        
        # Load line profiles, templates and calculate nail coordinates
        self.line_profiles = load_line_profiles_cached(self.resolution)
        self.templates = load_templates_cached()
        self.nail_coords = self.create_circle_nail_positions(self.num_nails, self.resolution // 2)
    
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
        
    def create_circle_nail_positions(self, num_nails, radius_px):
        nails = []
        for i in range(num_nails):
            theta = 2 * np.pi * i / num_nails
            y = int(radius_px * (1 + np.sin(theta)))
            x = int(radius_px * (1 + np.cos(theta)))
            y = np.clip(y, 0, self.resolution - 1)
            x = np.clip(x, 0, self.resolution - 1)
            nails.append((y, x))
        return nails
    
    def compute_string_length_km(
        self,
        sequence,
    ):
        nail_circle_diameter_mm = self.board_diameter_mm
        mm_per_pixel = nail_circle_diameter_mm / self.resolution
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
    
    def candidate_nails_array(self):
        # Helper for numba
        max_len = max(len(row) for row in self.candidate_nails)
        arr = np.full((self.num_nails, max_len), -1, dtype=np.int32)
        for i, row in enumerate(self.candidate_nails):
            arr[i, :len(row)] = row
        return arr

    # --------------------------------------------------
    # Core algorithm
    # --------------------------------------------------

    def preprocess(self, image_rgb: Image.Image):
        image_rgb = image_rgb.convert("RGB")
        
        # 1. Background removal (must come first to work effectively)
        if self.background_removal:
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
        
    def generate_sequence(self, target):
        """
        If possible, use generate_sequence_numba instead (5x faster and same sequence)
        """
        residual = np.ones_like(target, dtype=np.float32) - target # white canvas minus target

        current = 0
        path = []
        candidate_nails = self.candidate_nails
        line_strength = self.line_strength
        line_profiles = self.line_profiles

        while True:
            best = None
            best_imp = 0.0

            for i in candidate_nails[current]:
                line_id = line_profiles["map"][current, i]
                if line_id < 0:
                    continue

                start = line_profiles["start"][line_id]
                end   = line_profiles["end"][line_id]

                rr  = line_profiles["rr"][start:end]
                cc  = line_profiles["cc"][start:end]
                val = line_profiles["val"][start:end]

                # improvement formula
                # before = canvas[rr, cc] 
                # after = before - line_strength * val
                # target_slice = target[rr,cc]
                # imp = np.sum((before - target_slice) ** 2 - (after - target_slice) ** 2)
                
                # simplifies!
                # sub in r = before - target_slice
                # imp = sum(r^2 - (r - line_strength*val)^2) 
                #     = sum(2*r*line_strength*val - (line_strength*val)^2)
                #     = 2*line_strength*sum(r*val) - line_strength^2 * sum(val^2)
                r = residual[rr, cc]  # residual along the line profile
                dot_r_val = np.dot(r, val) # sum(r * val)
                sum_val_sq = np.dot(val, val) # sum(val^2)
                imp = 2 * line_strength * dot_r_val - (line_strength**2) * sum_val_sq

                if imp > best_imp:
                    best_imp = imp
                    best = i

            if best is None or len(path) >= self.max_lines:
                break
            
            # Update the residual with the line
            line_id = line_profiles["map"][current, best]
            start = line_profiles["start"][line_id]
            end   = line_profiles["end"][line_id]

            rr  = line_profiles["rr"][start:end]
            cc  = line_profiles["cc"][start:end]
            val = line_profiles["val"][start:end]

            residual[rr, cc] -= np.clip(line_strength * val,-1,1)

            path.append(best)
            current = best

        return path

    def generate_sequence_numba(self, target, use_importance = False, importance = 0):
        """
        Numba compiled string art generator (~1s).
        """
        candidate_arr = self.candidate_nails_array()
        if use_importance == False:
            importance = np.ones_like(target, dtype=np.float32) # safe deafult

        seq = generate_sequence_numba_func(
            target,
            use_importance,
            importance, 
            candidate_arr,
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

    def circular_crop_rgba(self, image, bgcolor=(0, 0, 0, 0)):
        """
        Input and output are PIL images
        """
        w, h = image.size
        assert w == h, "Image must be square for circular crop"
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, w, h), fill=255)
        image = image.convert("RGBA")
        result = Image.new("RGBA", (w, h), bgcolor)
        result.paste(image, (0, 0), mask)
        return result

    def place_on_template(self, template, coords, art, feather_px=3):
        """
        Place art on template with blurred edges for a seamless transition
        """
        (cx, cy), diameter = coords
        art_resized = art.resize((diameter, diameter), Image.LANCZOS).convert("RGBA")
        mask = Image.new("L", (diameter, diameter), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (
                feather_px,
                feather_px,
                diameter - feather_px,
                diameter - feather_px,
            ),
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(feather_px))
        art_resized.putalpha(mask)
        x = int(cx - diameter / 2)
        y = int(cy - diameter / 2)
        composite = template.copy()
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
        # Define circle coordinates (center (x,y from top left), diameter)
        template_wall_coords    = ([1515, 2210], 1225)

        LIGHTING = {
            "wall":    {"brightness": 0.75, "tint": (1.05, 1.02, 0.95)},
        }

        plain_render = self.render_from_sequence(sequence,self.nail_coords,thread_type=self.thread_type)

        # Helper to apply lighting based on preset name
        def render_with_lighting(name):
            preset = LIGHTING.get(name, {})
            return self.apply_lighting(
                plain_render,
                brightness=preset.get("brightness", 1.0),
                tint=preset.get("tint", (1.0, 1.0, 1.0))
            )

        previews = {
            "wall":    self.place_on_template(self.templates["wall"], template_wall_coords, render_with_lighting("wall")),
        }

        return previews, plain_render
         
    def render_from_sequence(
            self,
            sequence,
            nail_positions,
            thread_type="nylon",
            render_resolution=1000,     
            board_diameter_mm=480,
            supersample = 3, # 2 (good) or 3 (best)
            jitter_mm=2,            
        ):
        """     
        2-3s
        Render a high-quality string art preview from a nail sequence.
        Uses line_aa and a 0-1 float canvas which enables control of darkness by line_strength.
        Larger resolution than generation canvas and has jitter.
        """
        
        W = render_resolution * supersample
        mm_2_px = W / board_diameter_mm
        
        # Determine correct line strength for the thread type, resolution and board diameter
        if thread_type == "nylon": 
            line_strength = self.kNylon * W/500 * 480/board_diameter_mm
        elif thread_type == "poly": 
            line_strength = self.kPoly * W/500 * 480/board_diameter_mm
        else:
            raise ValueError("Invalid thread type")

        # Background canvas (float [0,1])
        canvas = np.ones((W, W), dtype=np.float32)

        # Convert nail pixel coords -> supersampled float coords
        def to_px_float(yx):
            y, x = yx
            return (
                (x + 0.5) * (W / self.resolution),
                (y + 0.5) * (W / self.resolution)
            )

        # Jitter in pixels (convert mm → pixels)
        max_jitter_px = jitter_mm * mm_2_px
        def jitter(p):
            return (
                p[0] + random.uniform(-max_jitter_px, max_jitter_px),
                p[1] + random.uniform(-max_jitter_px, max_jitter_px)
            )

        prev_idx = sequence[0]
        for idx in sequence[1:]:
            p0 = to_px_float(nail_positions[prev_idx])
            p1 = to_px_float(nail_positions[idx])

            # Apply sub-pixel jitter
            p0_j = jitter(p0)
            p1_j = jitter(p1)

            # Convert to integer pixel coordinates for line_aa
            y0, x0 = int(round(p0_j[1])), int(round(p0_j[0]))
            y1, x1 = int(round(p1_j[1])), int(round(p1_j[0]))

            rr, cc, val = line_aa(y0, x0, y1, x1)

            # Clip to canvas bounds (necessary with jitter)
            valid = (
                (rr >= 0) & (rr < W) &
                (cc >= 0) & (cc < W)
            )
            rr = rr[valid]
            cc = cc[valid]
            val = val[valid]

            # Apply the line
            canvas[rr, cc] = np.clip(canvas[rr, cc] - line_strength * val, 0, 1)

            prev_idx = idx

        # Convert canvas back to PIL image
        img = Image.fromarray((canvas * 255).astype(np.uint8)).convert("RGB")
        
        # Downsample for anti-aliasing
        if supersample > 1:
            img = img.resize(
                (render_resolution, render_resolution),
                resample=Image.LANCZOS
            )

        img = self.circular_crop_rgba(img)
        return img
