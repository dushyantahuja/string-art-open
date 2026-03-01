import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.draw import line_aa, disk
from PIL import Image, ImageDraw
from typing import Iterable, List
import io
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

class StringArtUtils():
    """
    Utility functions for string art process
    i.e. anything not in StringArtEngine
    """
    def __init__(self):  
       pass
   
    @staticmethod
    def create_nail_positions(num_nails, diameter_px, pattern):
        """
        Generate nail positions for different patterns: circular or square
        Uses a very small margin to inset the nails from edge
        """
        nails = []

        rMargin = 0.993
        inset_px = (1.0 - rMargin) * diameter_px / 2

        inner_diameter = round(diameter_px * rMargin)
        inner_radius = inner_diameter / 2

        if pattern == 'circle':
            for i in range(num_nails):
                theta = 2 * np.pi * i / num_nails

                # Generate in inner coordinate system
                y = inner_radius * (1 + np.sin(theta))
                x = inner_radius * (1 + np.cos(theta))

                # Recenter into full canvas
                y += inset_px
                x += inset_px

                nails.append((int(round(y)), int(round(x))))

            return nails

        elif pattern == 'square':
            side = inner_diameter - 1
            perimeter = 4 * side

            for i in range(num_nails):
                p = int(i * perimeter / num_nails)

                if p < side:                      # top
                    y, x = 0, p
                elif p < 2 * side:                # right
                    y, x = p - side, side
                elif p < 3 * side:                # bottom
                    y, x = side, side - (p - 2 * side)
                else:                             # left
                    y, x = side - (p - 3 * side), 0

                # Recenter
                y += inset_px
                x += inset_px

                nails.append((int(round(y)), int(round(x))))

            return nails

        else:
            raise Exception("Invalid pattern")
    
    @staticmethod
    def get_candidate_nails(nail_coords, num_nails, min_distance, pattern):
        """
        Precompute valid candidate nails for each nail index.

        Constraints:
        1) minimum index distance with wraparound
        2) geometry constraint:
            - circle: none
            - square: cannot connect nails on the same side

        Returns:
            int32 array of shape (num_nails, max_candidates)
            padded with -1 (Numba compatible)
        """

        nail_coords = np.asarray(nail_coords, dtype=np.int32)

        # --- classify square side for each nail ---
        # sides: 0=top, 1=right, 2=bottom, 3=left
        if pattern == "square":
            min_y = nail_coords[:, 0].min()
            max_y = nail_coords[:, 0].max()
            min_x = nail_coords[:, 1].min()
            max_x = nail_coords[:, 1].max()

            side_of = np.empty(num_nails, dtype=np.int8)
            for i, (y, x) in enumerate(nail_coords):
                if y == min_y:
                    side_of[i] = 0
                elif x == max_x:
                    side_of[i] = 1
                elif y == max_y:
                    side_of[i] = 2
                else:
                    side_of[i] = 3
        else:
            side_of = None  # unused for circle

        candidate_nails = []

        for i in range(num_nails):
            row = []

            for j in range(num_nails):
                if i == j:
                    continue

                # --- constraint 1: min index distance with wraparound ---
                d = abs(i - j)
                if min(d, num_nails - d) <= min_distance:
                    continue

                # --- constraint 2: square side rule ---
                if pattern == "square":
                    if side_of[i] == side_of[j]:
                        continue

                row.append(j)

            candidate_nails.append(row)

        # --- convert to fixed-size NumPy array (Numba-safe) ---
        max_len = max(len(r) for r in candidate_nails)
        arr = np.full((num_nails, max_len), -1, dtype=np.int32)

        for i, row in enumerate(candidate_nails):
            arr[i, :len(row)] = row

        return arr
        
    @staticmethod 
    def precompute_line_profiles(nail_coords):
        """
        Line profiles with no python objects, smaller and suitable for numba compiler
        Works with any pattern
        """
        num_nails = len(nail_coords)

        # Calculate valid pairs on nail indices without repetition
        valid_pairs = []
        for i in range(num_nails):
            for j in range(i+1, num_nails):
                valid_pairs.append((i, j))

        # Precompute all lines and count total pixels
        total_pixels = 0
        for (i, j) in valid_pairs:
            yi, xi = nail_coords[i]
            yj, xj = nail_coords[j]
            rr, cc, val = line_aa(yi, xi, yj, xj)
            total_pixels += len(rr)

        # Allocate arrays
        line_rr = np.empty(total_pixels, dtype=np.int32)
        line_cc = np.empty(total_pixels, dtype=np.int32)
        line_val = np.empty(total_pixels, dtype=np.float32)

        line_start = np.empty(len(valid_pairs), dtype=np.int32)
        line_end   = np.empty(len(valid_pairs), dtype=np.int32)

        line_idx_map = -np.ones((num_nails, num_nails), dtype=np.int32)

        # Fill arrays
        idx = 0
        line_id = 0
        for (i, j) in valid_pairs:
            yi, xi = nail_coords[i]
            yj, xj = nail_coords[j]
            rr, cc, val = line_aa(yi, xi, yj, xj)

            L = len(rr)
            line_start[line_id] = idx
            line_end[line_id] = idx + L

            line_rr[idx:idx+L] = rr
            line_cc[idx:idx+L] = cc
            line_val[idx:idx+L] = val.astype(np.float32)

            line_idx_map[i, j] = line_id
            line_idx_map[j, i] = line_id

            idx += L
            line_id += 1

        return {
            "rr": line_rr,
            "cc": line_cc,
            "val": line_val,
            "start": line_start,
            "end": line_end,
            "map": line_idx_map
        }

    @staticmethod
    def make_template(nail_coords):
        """
        Generate printable nail placement template for arbitrary shapes.

        nail_coords: list of (row, col) in generation resolution (e.g. 500x500)
        """
        RESOLUTION = 3000  # DON'T CHANGE
        PADDING = 70 # DON'T CHANGE
        NAIL_RADIUS_VISUAL = 6

        canvas = np.ones((RESOLUTION, RESOLUTION), dtype=np.float32)

        # --- convert input nail coords to numpy ---
        nail_coords = np.asarray(nail_coords, dtype=np.float32)

        # --- determine bounding box in source space ---
        min_r, min_c = nail_coords.min(axis=0)
        max_r, max_c = nail_coords.max(axis=0)

        src_h = max_r - min_r
        src_w = max_c - min_c
        src_size = max(src_h, src_w)

        # --- scale to fit nicely inside template ---
        target_size = RESOLUTION - 2 * PADDING
        scale = target_size / src_size

        # --- center in destination ---
        dst_center = RESOLUTION / 2
        src_center_r = (min_r + max_r) / 2
        src_center_c = (min_c + max_c) / 2

        nail_coords_scaled = []
        for r, c in nail_coords:
            r_s = (r - src_center_r) * scale + dst_center
            c_s = (c - src_center_c) * scale + dst_center
            nail_coords_scaled.append((int(round(r_s)), int(round(c_s))))

        nail_coords_scaled = np.asarray(nail_coords_scaled)

        # --- draw nails ---
        for r_nail, c_nail in nail_coords_scaled:
            rr, cc = disk((r_nail, c_nail), NAIL_RADIUS_VISUAL, shape=canvas.shape)
            canvas[rr, cc] = 0

        dpi = 200
        h, w = canvas.shape
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(canvas, cmap="gray", interpolation="nearest")
        ax.axis("off")

        # --- centroid for label direction ---
        center_r = nail_coords_scaled[:, 0].mean()
        center_c = nail_coords_scaled[:, 1].mean()

        # --- label nails ---
        for idx, (r_nail, c_nail) in enumerate(nail_coords_scaled):
            dr = center_r - r_nail
            dc = center_c - c_nail

            length = np.hypot(dr, dc)
            if length == 0:
                continue

            dr /= length
            dc /= length

            offset = NAIL_RADIUS_VISUAL + 40
            r_text = r_nail - dr * offset
            c_text = c_nail - dc * offset

            # connector line
            ax.plot(
                [c_nail, c_text],
                [r_nail, r_text],
                color="black",
                linewidth=0.2,
            )

            # label style rules (unchanged)
            if idx % 50 == 0:
                fontweight = "bold"
                fontsize = 10
            elif idx % 10 == 0:
                fontweight = "bold"
                fontsize = 8
            else:
                fontweight = "normal"
                fontsize = 6

            ax.text(
                c_text,
                r_text,
                str(idx),
                color="black",
                fontsize=fontsize,
                fontweight=fontweight,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    pad=0.4,
                ),
            )

        fig.savefig("canvas_with_labels.png", dpi=200)
        plt.close(fig)
    
    @staticmethod
    def draw_importance_mask(target: np.ndarray, importance=None, max_val = 5.0) -> np.ndarray:
        """
        Interactive importance mask editor using matplotlib.

        target: float32 array, range [0,1]
        importance: optional float32 array, same shape as target
        max_val: maximum value for mask (5 seems okay)
        returns: float32 importance mask, range [1.0, 1.2]
        """

        assert target.ndim == 2
        h, w = target.shape

        # Initialize importance to 1 or use supplied
        if importance is None:
            importance = np.ones((h, w), dtype=np.float32)
        
        fig, ax = plt.subplots()
        painting = {"mode": +1}  # +1 = paint, -1 = erase

        def mode_str():
            return "PAINT" if painting["mode"] > 0 else "ERASE"

        fig.suptitle(
            f"Importance Mask Editor - Mode: {mode_str()}\n"
            "'b' toggle paint/erase | 'c' clear | Scroll=brush size | SPACE/Esc to accept"
        )

        ax.imshow(target, cmap="gray", vmin=0, vmax=1)
        overlay = ax.imshow(importance, cmap="Reds", alpha=0.5, vmin=1, vmax=max_val)

        # Brush preview inset
        inset_ax = fig.add_axes([0.8, 0.8, 0.1, 0.1])
        brush_radius = 20
        brush_preview_circle = Circle((0.5, 0.5), 0.3, color="red", alpha=0.5)
        inset_ax.add_patch(brush_preview_circle)
        inset_ax.set_xlim(0,1)
        inset_ax.set_ylim(0,1)
        inset_ax.axis("off")

        brush_strength = (max_val - 1) * 0.5 # fraction of range

        def get_brush_mask(radius):
            yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
            return np.clip(1.0 - np.sqrt(xx**2 + yy**2)/radius, 0.0, 1.0)
        brush_mask = get_brush_mask(brush_radius)

        def redraw_overlay():
            overlay.set_data(importance)
            fig.canvas.draw_idle()

        def update_brush_preview():
            brush_preview_circle.set_radius(brush_radius / max(h, w))
            inset_ax.draw_artist(brush_preview_circle)
            fig.canvas.draw_idle()

        def apply_brush(x, y, sign):
            x = int(x)
            y = int(y)
            if x < 0 or y < 0 or x >= w or y >= h:
                return

            y0 = max(0, y - brush_radius)
            x0 = max(0, x - brush_radius)
            y1 = min(h, y + brush_radius + 1)
            x1 = min(w, x + brush_radius + 1)

            my0 = brush_radius - (y - y0)
            mx0 = brush_radius - (x - x0)
            my1 = my0 + (y1 - y0)
            mx1 = mx0 + (x1 - x0)

            importance[y0:y1, x0:x1] += sign * brush_strength * brush_mask[my0:my1, mx0:mx1]
            np.clip(importance, 1, max_val, out=importance)
            redraw_overlay()

        def on_press(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                apply_brush(event.xdata, event.ydata, painting["mode"])

        def on_motion(event):
            if event.inaxes != ax or event.button != 1:
                return
            apply_brush(event.xdata, event.ydata, painting["mode"])

        def on_scroll(event):
            nonlocal brush_radius, brush_mask
            if event.button == "up":
                brush_radius = min(64, brush_radius + 2)
            else:
                brush_radius = max(2, brush_radius - 2)
            brush_mask = get_brush_mask(brush_radius)
            update_brush_preview()

        def on_key(event):
            nonlocal importance
            if event.key == "b":
                painting["mode"] *= -1
                fig.suptitle(
                    f"Importance Mask Editor - Mode: {mode_str()}\n"
                    "'b' toggle paint/erase | 'c' clear | Scroll=brush size | SPACE/Esc to accept"
                )
                fig.canvas.draw_idle()
            elif event.key == "c":
                importance[:] = 1.0
                redraw_overlay()
            elif event.key in [" ", "escape"]:
                ax.set_axis_off()
                fig.canvas.draw()
                bbox = ax.get_tightbbox(fig.canvas.get_renderer())
                bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

                fig.savefig(
                    "results/recent/importance.jpeg",
                    format="jpeg",
                    dpi=300,
                    bbox_inches=bbox,
                    pad_inches=0
                )
                plt.close("all")

        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("scroll_event", on_scroll)
        fig.canvas.mpl_connect("key_press_event", on_key)

        # If a previous importance mask is supplied, draw immediately
        redraw_overlay()
        update_brush_preview()
        plt.show()

        return importance.astype(np.float32)
    
    @staticmethod
    def circular_crop_rgba(image, bgcolor=(0, 0, 0, 0)):
        """
        Input and output are PIL images
        """
        w, h = image.size
        assert w == h, "Image must be square for circular crop (use largest square first)"
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, w, h), fill=255)
        image = image.convert("RGBA")
        result = Image.new("RGBA", (w, h), bgcolor)
        result.paste(image, (0, 0), mask)
        return result
    
    @staticmethod
    def build_sequence_pdf(sequence: Iterable[int], out_path=None) -> bytes:
        milestone_headers = {
        0: ["Getting Started..."],
        500: ["In the Swing of Things"],
        1000: ["This is fun!"],
        1500: ["Can you see your image yet?"],
        2000: ["Maybe halfway?"],
        2500: ["Don't give up!"],
        3000: ["So close..."],
        3500: ["Endgame"],
        4000: ["If you made it this far - well done"],
        4500: ["Thats a lot of lines, most people won't see this"],
        5000: ["Why does your string art have 5000 lines - is it a black image!?"]
        }
        closing_message = "Congrats on finishing your StringBoard! Please send me a photo! info@stringboard.co.uk"
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=LETTER)
        width, height = LETTER

        numbers: List[int] = list(sequence)
        milestone_headers = milestone_headers or {}

        margin_x = 50
        margin_y = 50
        line_height = 14

        y = height - margin_y

        def ensure_space(lines: int):
            nonlocal y
            required = lines * line_height
            if y < margin_y + required:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - margin_y

        for block_start in range(0, len(numbers), 100):
            block = numbers[block_start:block_start + 100]

            # Main block header
            ensure_space(3)
            c.setFont("Helvetica-Bold", 12)
            header = f"Lines {block_start + 1} – {block_start + len(block)}"
            c.drawString(margin_x, y, header)
            y -= 2 * line_height

            # Optional milestone headers
            if block_start in milestone_headers:
                c.setFont("Helvetica-Oblique", 10)
                for text in milestone_headers[block_start]:
                    ensure_space(1)
                    c.drawString(margin_x, y, text)
                    y -= line_height
                y -= line_height / 2

            # Number rows: 5 × 20
            c.setFont("Helvetica", 10)
            for row_start in range(0, len(block), 20):
                ensure_space(1)
                row = block[row_start:row_start + 20]
                line = ", ".join(map(str, row))
                c.drawString(margin_x, y, line)
                y -= line_height

            y -= line_height  # spacing after block
        
        c.setFont("Helvetica-Oblique", 10)
        
        # Ensure space for message
        if y < margin_y + line_height:
            c.showPage()
            c.setFont("Helvetica-Oblique", 10)
            y = height - margin_y

        c.drawString(margin_x, y, closing_message)
        y -= line_height

        c.save()
        
        if out_path is not None: # optionally save pdf directly
            with open(out_path, "wb") as f:
                f.write(buffer.getvalue())
            
        buffer.seek(0)
        return buffer.read()