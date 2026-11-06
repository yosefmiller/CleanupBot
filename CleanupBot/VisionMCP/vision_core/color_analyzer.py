"""CSS3 perceptual color analyzer for robotics obstacle & clutter identification."""

from __future__ import annotations

import logging
from typing import Any
import cv2
import numpy as np
from scipy.spatial import KDTree

log = logging.getLogger("vision_core.color_analyzer")

# 148 standard W3C / CSS3 named colors with exact RGB hex values
CSS3_COLORS: dict[str, tuple[int, int, int]] = {
    "aliceblue": (240, 248, 255),
    "antiquewhite": (250, 235, 215),
    "aqua": (0, 255, 255),
    "aquamarine": (127, 255, 212),
    "azure": (240, 255, 255),
    "beige": (245, 245, 220),
    "bisque": (255, 228, 196),
    "black": (0, 0, 0),
    "blanchedalmond": (255, 235, 205),
    "blue": (0, 0, 255),
    "blueviolet": (138, 43, 226),
    "brown": (165, 42, 42),
    "burlywood": (222, 184, 135),
    "cadetblue": (95, 158, 160),
    "chartreuse": (127, 255, 0),
    "chocolate": (210, 105, 30),
    "coral": (255, 127, 80),
    "cornflowerblue": (100, 149, 237),
    "cornsilk": (255, 248, 220),
    "crimson": (220, 20, 60),
    "cyan": (0, 255, 255),
    "darkblue": (0, 0, 139),
    "darkcyan": (0, 139, 139),
    "darkgoldenrod": (184, 134, 11),
    "darkgray": (169, 169, 169),
    "darkgreen": (0, 100, 0),
    "darkgrey": (169, 169, 169),
    "darkkhaki": (189, 183, 107),
    "darkmagenta": (139, 0, 139),
    "darkolivegreen": (85, 107, 47),
    "darkorange": (255, 140, 0),
    "darkorchid": (153, 50, 204),
    "darkred": (139, 0, 0),
    "darksalmon": (233, 150, 122),
    "darkseagreen": (143, 188, 143),
    "darkslateblue": (72, 61, 139),
    "darkslategray": (47, 79, 79),
    "darkslategrey": (47, 79, 79),
    "darkturquoise": (0, 206, 209),
    "darkviolet": (148, 0, 211),
    "deeppink": (255, 20, 147),
    "deepskyblue": (0, 191, 255),
    "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105),
    "dodgerblue": (30, 144, 255),
    "firebrick": (178, 34, 34),
    "floralwhite": (255, 250, 240),
    "forestgreen": (34, 139, 34),
    "fuchsia": (255, 0, 255),
    "gainsboro": (220, 220, 220),
    "ghostwhite": (248, 248, 255),
    "gold": (255, 215, 0),
    "goldenrod": (218, 165, 32),
    "gray": (128, 128, 128),
    "green": (0, 128, 0),
    "greenyellow": (173, 255, 47),
    "grey": (128, 128, 128),
    "honeydew": (240, 255, 240),
    "hotpink": (255, 105, 180),
    "indianred": (205, 92, 92),
    "indigo": (75, 0, 130),
    "ivory": (255, 255, 240),
    "khaki": (240, 230, 140),
    "lavender": (230, 230, 250),
    "lavenderblush": (255, 240, 245),
    "lawngreen": (124, 252, 0),
    "lemonchiffon": (255, 250, 205),
    "lightblue": (173, 216, 230),
    "lightcoral": (240, 128, 128),
    "lightcyan": (224, 255, 255),
    "lightgoldenrodyellow": (250, 250, 210),
    "lightgray": (211, 211, 211),
    "lightgreen": (144, 238, 144),
    "lightgrey": (211, 211, 211),
    "lightpink": (255, 182, 193),
    "lightsalmon": (255, 160, 122),
    "lightseagreen": (32, 178, 170),
    "lightskyblue": (135, 206, 250),
    "lightslategray": (119, 136, 153),
    "lightslategrey": (119, 136, 153),
    "lightsteelblue": (176, 196, 222),
    "lightyellow": (255, 255, 224),
    "lime": (0, 255, 0),
    "limegreen": (50, 205, 50),
    "linen": (250, 240, 230),
    "magenta": (255, 0, 255),
    "maroon": (128, 0, 0),
    "mediumaquamarine": (102, 205, 170),
    "mediumblue": (0, 0, 205),
    "mediumorchid": (186, 85, 211),
    "mediumpurple": (147, 112, 219),
    "mediumseagreen": (60, 179, 113),
    "mediumslateblue": (123, 104, 238),
    "mediumspringgreen": (0, 250, 154),
    "mediumturquoise": (72, 209, 204),
    "mediumvioletred": (199, 21, 133),
    "midnightblue": (25, 25, 112),
    "mintcream": (245, 255, 250),
    "mistyrose": (255, 228, 225),
    "moccasin": (255, 228, 181),
    "navajowhite": (255, 222, 173),
    "navy": (0, 0, 128),
    "oldlace": (253, 245, 230),
    "olive": (128, 128, 0),
    "olivedrab": (107, 142, 35),
    "orange": (255, 165, 0),
    "orangered": (255, 69, 0),
    "orchid": (218, 112, 214),
    "palegoldenrod": (238, 232, 170),
    "palegreen": (152, 251, 152),
    "paleturquoise": (175, 238, 238),
    "palevioletred": (219, 112, 147),
    "papayawhip": (255, 239, 213),
    "peachpuff": (255, 218, 185),
    "peru": (205, 133, 63),
    "pink": (255, 192, 203),
    "plum": (221, 160, 221),
    "powderblue": (176, 224, 230),
    "purple": (128, 0, 128),
    "rebeccapurple": (102, 51, 153),
    "red": (255, 0, 0),
    "rosybrown": (188, 143, 143),
    "royalblue": (65, 105, 225),
    "saddlebrown": (139, 69, 19),
    "salmon": (250, 128, 114),
    "sandybrown": (244, 164, 96),
    "seagreen": (46, 139, 87),
    "seashell": (255, 245, 238),
    "sienna": (160, 82, 45),
    "silver": (192, 192, 192),
    "skyblue": (135, 206, 235),
    "slateblue": (106, 90, 205),
    "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144),
    "snow": (255, 250, 250),
    "springgreen": (0, 255, 127),
    "steelblue": (70, 130, 180),
    "tan": (210, 180, 140),
    "teal": (0, 128, 128),
    "thistle": (216, 191, 216),
    "tomato": (255, 99, 71),
    "turquoise": (64, 224, 208),
    "violet": (238, 130, 238),
    "wheat": (245, 222, 179),
    "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245),
    "yellow": (255, 255, 0),
    "yellowgreen": (154, 205, 50),
}

# Mapping of CSS3 specific names to broad intuitive color families
COLOR_FAMILY_MAP: dict[str, str] = {
    # Red
    "red": "red", "crimson": "red", "darkred": "red", "firebrick": "red",
    "indianred": "red", "salmon": "red", "darksalmon": "red", "lightcoral": "red",
    "maroon": "red", "brown": "red", "sienna": "red", "rosybrown": "red",
    # Orange
    "orange": "orange", "darkorange": "orange", "coral": "orange", "tomato": "orange",
    "orangered": "orange", "sandybrown": "orange", "chocolate": "orange",
    "peru": "orange", "saddlebrown": "orange", "burlywood": "orange", "tan": "orange",
    "peachpuff": "orange", "bisque": "orange", "navajowhite": "orange", "moccasin": "orange",
    # Yellow
    "yellow": "yellow", "gold": "yellow", "goldenrod": "yellow",
    "darkgoldenrod": "yellow", "khaki": "yellow", "palegoldenrod": "yellow",
    "darkkhaki": "yellow",
    "lemonchiffon": "yellow", "lightgoldenrodyellow": "yellow", "lightyellow": "yellow",
    "papayawhip": "yellow", "wheat": "yellow", "cornsilk": "yellow", "blanchedalmond": "yellow",
    # Green
    "green": "green", "darkgreen": "green", "forestgreen": "green", "lime": "green",
    "limegreen": "green", "lightgreen": "green", "palegreen": "green", "seagreen": "green",
    "mediumseagreen": "green", "springgreen": "green", "mediumspringgreen": "green",
    "olive": "green", "olivedrab": "green", "darkolivegreen": "green", "yellowgreen": "green",
    "lawngreen": "green", "chartreuse": "green", "darkseagreen": "green",
    "greenyellow": "green",
    # Cyan / Teal
    "cyan": "cyan", "aqua": "cyan", "darkcyan": "cyan", "teal": "teal",
    "mediumturquoise": "cyan", "turquoise": "cyan", "paleturquoise": "cyan",
    "aquamarine": "cyan", "lightseagreen": "cyan", "cadetblue": "cyan",
    "darkturquoise": "cyan", "mediumaquamarine": "cyan", "lightcyan": "cyan",
    # Blue
    "blue": "blue", "mediumblue": "blue", "darkblue": "blue", "navy": "blue",
    "midnightblue": "blue", "royalblue": "blue", "cornflowerblue": "blue",
    "dodgerblue": "blue", "deepskyblue": "blue", "skyblue": "blue",
    "lightskyblue": "blue", "steelblue": "blue", "lightsteelblue": "blue",
    "powderblue": "blue", "lightblue": "blue",
    # Purple
    "purple": "purple", "darkmagenta": "purple", "magenta": "purple", "fuchsia": "purple",
    "orchid": "purple", "darkorchid": "purple", "mediumorchid": "purple",
    "violet": "purple", "plum": "purple", "thistle": "purple", "indigo": "purple",
    "darkslateblue": "purple", "slateblue": "purple", "mediumslateblue": "purple",
    "rebeccapurple": "purple", "blueviolet": "purple", "darkviolet": "purple",
    # Pink
    "pink": "pink", "lightpink": "pink", "hotpink": "pink", "deeppink": "pink",
    "mediumvioletred": "pink", "palevioletred": "pink",
    # White / Light neutral
    "white": "white", "snow": "white", "honeydew": "white", "mintcream": "white",
    "azure": "white", "aliceblue": "white", "ghostwhite": "white", "whitesmoke": "white",
    "seashell": "white", "beige": "white", "oldlace": "white", "floralwhite": "white",
    "ivory": "white", "antiquewhite": "white", "linen": "white", "lavender": "white",
    "lavenderblush": "white", "mistyrose": "white",
    # Gray / Neutral metallic
    "gray": "gray", "dimgray": "gray", "darkgray": "gray", "silver": "silver",
    "lightgray": "gray", "gainsboro": "gray", "slategray": "gray",
    "darkslategray": "gray", "lightslategray": "gray", "grey": "gray",
    "dimgrey": "gray", "darkgrey": "gray", "lightgrey": "gray",
    "slategrey": "gray", "lightslategrey": "gray",
    # Black / Dark
    "black": "black",
}


class ColorAnalyzer:
    """Fast, perceptual CIELAB KDTree color analyzer for floor hazards."""

    def __init__(self) -> None:
        self.color_names: list[str] = list(CSS3_COLORS.keys())
        
        # Build array of RGB values in shape (N, 1, 3) for cv2 RGB->Lab conversion
        rgb_arr = np.array([CSS3_COLORS[c] for c in self.color_names], dtype=np.uint8).reshape(-1, 1, 3)
        lab_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float32)
        
        # KDTree over 3D Lab coordinates for perceptual Delta E matching
        self.kdtree: KDTree = KDTree(lab_arr)

    def analyze_crop(self, bgr_crop: np.ndarray) -> dict[str, Any]:
        """
        Analyze the color of an object bounding box crop.
        Detects multicolored objects (e.g. hacky sack) vs single-colored items.
        
        Returns:
            {
                'color_name': str,       # e.g. 'darkorange', 'gold', 'sienna', 'multicolored'
                'color_family': str,     # e.g. 'orange', 'yellow', 'red', 'multicolored'
                'label': str,            # Color-first label: e.g. 'orange item', 'multicolored item'
                'is_multicolored': bool,
                'delta_e': float,
            }
        """
        if bgr_crop is None or bgr_crop.size == 0 or bgr_crop.shape[0] < 3 or bgr_crop.shape[1] < 3:
            return {
                "color_name": "unknown",
                "color_family": "object",
                "label": "floor item",
                "is_multicolored": False,
                "delta_e": 999.0,
            }

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Chromatic pixels: saturated enough and sufficiently lit (avoids floor shadow/carpet)
        chroma_mask = (s > 50) & (v > 40) & (v < 250)
        chroma_count = int(np.count_nonzero(chroma_mask))
        total_pixels = bgr_crop.shape[0] * bgr_crop.shape[1]
        chroma_ratio = chroma_count / max(total_pixels, 1)

        # 1. Multi-color check (e.g. hacky sack / sports ball with multi-hue patches)
        if chroma_ratio >= 0.20 and chroma_count >= 30:
            hues = h[chroma_mask]
            # 12 bins across 0-180 (15 degrees per bin)
            hist, _ = np.histogram(hues, bins=12, range=(0, 180))
            # Peaks with at least 15% of chromatic pixels
            sig_bins = np.where(hist >= 0.15 * chroma_count)[0]
            if len(sig_bins) >= 2:
                # Circular distance on 12-bin hue wheel (modulo 12)
                # Hue 0 and Hue 11 are both Red (adjacent across 0/180 wrap-around)
                diffs = [
                    min(abs(int(sig_bins[i]) - int(sig_bins[j])), 12 - abs(int(sig_bins[i]) - int(sig_bins[j])))
                    for i in range(len(sig_bins))
                    for j in range(i + 1, len(sig_bins))
                ]
                if any(d >= 2 for d in diffs):
                    return {
                        "color_name": "multicolored",
                        "color_family": "multicolored",
                        "label": "multicolored item",
                        "is_multicolored": True,
                        "delta_e": 0.0,
                    }

        # 2. Representative color extraction
        if chroma_ratio >= 0.06:
            # Use chromatic pixels if sufficient color is present
            b_val = float(np.median(bgr_crop[:, :, 0][chroma_mask]))
            g_val = float(np.median(bgr_crop[:, :, 1][chroma_mask]))
            r_val = float(np.median(bgr_crop[:, :, 2][chroma_mask]))
        else:
            # Achromatic / neutral item (e.g. silver battery, gray earbuds case, white paper)
            # Sample central 60% of crop to avoid border floor background
            ch = bgr_crop.shape[0]
            cw = bgr_crop.shape[1]
            y0, y1 = max(0, int(ch * 0.2)), min(ch, int(ch * 0.8))
            x0, x1 = max(0, int(cw * 0.2)), min(cw, int(cw * 0.8))
            center_crop = bgr_crop[y0:y1, x0:x1] if (y1 > y0 and x1 > x0) else bgr_crop
            b_val = float(np.median(center_crop[:, :, 0]))
            g_val = float(np.median(center_crop[:, :, 1]))
            r_val = float(np.median(center_crop[:, :, 2]))

        # Convert representative BGR pixel to CIELAB
        rep_pixel = np.array([[[b_val, g_val, r_val]]], dtype=np.uint8)
        lab_pixel = cv2.cvtColor(rep_pixel, cv2.COLOR_BGR2Lab).reshape(3).astype(np.float32)

        # Query KDTree for nearest CSS3 color
        dist, idx = self.kdtree.query(lab_pixel, k=1)
        css3_name = self.color_names[idx]
        color_family = COLOR_FAMILY_MAP.get(css3_name, css3_name)

        # Color-first label (e.g. 'orange item', 'yellow item', 'silver item')
        label = f"{color_family} item"

        return {
            "color_name": css3_name,
            "color_family": color_family,
            "label": label,
            "is_multicolored": False,
            "delta_e": round(float(dist), 1),
        }
