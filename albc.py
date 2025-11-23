# app.py
import streamlit as st
import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

st.set_page_config(page_title="Thermal ↔ RGB Alignment", layout="wide")

st.title("Thermal ↔ RGB Alignment and Overlay")
st.write("Automatically crop black borders from RGB, align thermal to RGB, and create overlays.")

# --- Helpers ---
def crop_black_borders_rgb(img_bgr, tol=8):
    """
    Detect black borders in an RGB image and crop them out.
    tol: threshold for considering a pixel 'black' (0-255)
    Returns cropped image and bounding box (x,y,w,h).
    """
    if img_bgr is None:
        return None, (0, 0, img_bgr.shape[1], img_bgr.shape[0])
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # mask of non-black pixels
    mask = gray > tol
    if mask.sum() == 0:
        # image completely black (unexpected). return original
        h, w = img_bgr.shape[:2]
        return img_bgr, (0, 0, w, h)
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at top
    cropped = img_bgr[y0:y1, x0:x1]
    return cropped, (x0, y0, x1 - x0, y1 - y0)

def resize_to(img, target_shape):
    """
    Resize img to target_shape which is (height, width).
    """
    h, w = target_shape
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

def align_thermal_to_rgb(thermal_gray, rgb_gray, max_features=2000, good_match_percent=0.15):
    """
    Align thermal_gray to rgb_gray using ORB feature detection + homography.
    Returns warped_thermal (same size as rgb_gray) and the homography matrix (or None).
    """
    # ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(thermal_gray, None)  # thermal
    kp2, des2 = orb.detectAndCompute(rgb_gray, None)      # rgb (cropped and resized)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None

    # Match features.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # Sort by distance. Best first.
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only good matches
    num_good = max(4, int(len(matches) * good_match_percent))
    good_matches = matches[:num_good]

    if len(good_matches) < 4:
        return None, None

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Compute homography
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H, mask

def apply_colormap_to_thermal(warped_thermal_gray):
    """Map thermal grayscale to colored BGR using a colormap."""
    normalized = cv2.normalize(warped_thermal_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)  # BGR
    return colored

def overlay_images(rgb_bgr, thermal_color_bgr, alpha=0.5):
    """
    Blend rgb_bgr and thermal_color_bgr using addWeighted.
    Both must be same shape and BGR.
    """
    return cv2.addWeighted(rgb_bgr, 1 - alpha, thermal_color_bgr, alpha, 0)

# --- Streamlit UI ---
st.sidebar.header("Settings")
input_folder = st.sidebar.text_input("Input folder (contains pairs)", value="D:/assignments/thermal_image/images")
output_folder = st.sidebar.text_input("Output folder", value=os.path.join(input_folder, "aligned_output"))
alpha = st.sidebar.slider("Overlay alpha (thermal)", 0.0, 1.0, 0.4, 0.05)
tol = st.sidebar.slider("Black border tolerance (0-50)", 0, 50, 8, 1)
good_match_percent = st.sidebar.slider("Good match percent for features", 1, 50, 15, 1) / 100.0
max_features = st.sidebar.number_input("Max ORB features", value=2000, min_value=500, max_value=5000, step=100)

st.sidebar.markdown("**Instructions**: Make sure images follow naming `XXXX_T.JPG` and `XXXX_Z.JPG`.")

if not os.path.isdir(input_folder):
    st.error("Input folder does not exist. Please provide a valid path.")
else:
    os.makedirs(output_folder, exist_ok=True)

    if st.button("Start processing"):
        # gather pairs
        thermal_paths = glob.glob(os.path.join(input_folder, "*_T.JPG")) + glob.glob(os.path.join(input_folder, "*_T.jpg"))
        processed = 0
        failed = []
        if not thermal_paths:
            st.warning("No thermal images found in the input folder with suffix '_T.JPG' or '_T.jpg'.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, tpath in enumerate(tqdm(thermal_paths, desc="Pairs")):
                try:
                    base = os.path.basename(tpath)
                    # derive identifier XXXX by removing the suffix _T.JPG or _T.jpg
                    if base.endswith("_T.JPG") or base.endswith("_T.jpg"):
                        identifier = base[:-6]
                    elif base.endswith("_T.JPEG") or base.endswith("_T.jpeg"):
                        identifier = base.rsplit("_T.", 1)[0]
                    else:
                        identifier = base.rsplit("_T", 1)[0]

                    # corresponding RGB path (try .JPG and .jpg)
                    rgb_candidates = [
                        os.path.join(input_folder, f"{identifier}_Z.JPG"),
                        os.path.join(input_folder, f"{identifier}_Z.jpg"),
                        os.path.join(input_folder, f"{identifier}_Z.JPEG"),
                        os.path.join(input_folder, f"{identifier}_Z.jpeg"),
                    ]
                    rgb_path = next((p for p in rgb_candidates if os.path.exists(p)), None)
                    if rgb_path is None:
                        failed.append((identifier, "RGB not found"))
                        continue

                    # Read images
                    thermal = cv2.imread(tpath, cv2.IMREAD_UNCHANGED)  # thermal image (maybe grayscale or pseudo-color)
                    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                    if thermal is None or rgb is None:
                        failed.append((identifier, "read error"))
                        continue

                    # Convert thermal to grayscale if needed
                    if len(thermal.shape) == 3:
                        # if thermal has 3 channels, convert to gray
                        thermal_gray_orig = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
                    else:
                        thermal_gray_orig = thermal.copy()

                    # 1) Preprocess & crop RGB: detect black borders and remove
                    rgb_h, rgb_w = rgb.shape[:2]
                    cropped_rgb, bbox = crop_black_borders_rgb(rgb, tol=tol)
                    # Resize cropped RGB back to original full-screen size
                    cropped_resized_rgb = resize_to(cropped_rgb, (rgb_h, rgb_w))

                    # Prepare grayscale versions for alignment
                    rgb_gray_for_align = cv2.cvtColor(cropped_resized_rgb, cv2.COLOR_BGR2GRAY)

                    # 2) Align thermal image
                    # Resize thermal to approximate the rgb size to improve matching speed if it's very different
                    t_h, t_w = thermal_gray_orig.shape[:2]
                    # If thermal much smaller or larger, scale to rgb size first as an initial estimate
                    scale_factor = max(rgb_w / max(1, t_w), rgb_h / max(1, t_h))
                    # Resize thermal to roughly the same bounding area
                    if scale_factor != 1.0:
                        new_tw = int(round(t_w * scale_factor))
                        new_th = int(round(t_h * scale_factor))
                        thermal_gray_resized = cv2.resize(thermal_gray_orig, (new_tw, new_th), interpolation=cv2.INTER_LINEAR)
                    else:
                        thermal_gray_resized = thermal_gray_orig.copy()

                    H, mask = align_thermal_to_rgb(
                        thermal_gray_resized,
                        rgb_gray_for_align,
                        max_features=max_features,
                        good_match_percent=good_match_percent,
                    )

                    # If homography found, warp thermal to rgb size. If not, fallback to simple resize center placement.
                    if H is not None:
                        # We need to warp the thermal image (which was resized) into the RGB frame.
                        warped = cv2.warpPerspective(thermal_gray_resized, H, (rgb_w, rgb_h), flags=cv2.INTER_LINEAR)
                    else:
                        # fallback: scale thermal to rgb size
                        warped = cv2.resize(thermal_gray_resized, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)

                    # 3) Overlay requirements
                    thermal_color = apply_colormap_to_thermal(warped)
                    overlay_bgr = overlay_images(cropped_resized_rgb, thermal_color, alpha=alpha)

                    # 4) Save outputs
                    out_aligned_thermal = os.path.join(output_folder, f"{identifier}_aligned_thermal.jpg")
                    out_overlay = os.path.join(output_folder, f"{identifier}_overlay.jpg")
                    # Save aligned thermal as grayscale mapped to BGR (so it is a visible image)
                    cv2.imwrite(out_aligned_thermal, thermal_color)
                    # Save overlay (RGB remains visually same as cropped_resized_rgb plus thermal)
                    cv2.imwrite(out_overlay, overlay_bgr)

                    processed += 1

                    # update progress
                    progress_bar.progress((i + 1) / len(thermal_paths))
                    status_text.text(f"Processed: {processed}  — Last: {identifier}")

                except Exception as e:
                    failed.append((tpath, str(e)))
            progress_bar.progress(1.0)
            st.success(f"Done. Processed {processed} pairs. Failed: {len(failed)}")
            if failed:
                st.write("Failures (identifier, reason):")
                for f in failed[:20]:
                    st.write(f"- {f[0]} : {f[1]}")

            # Show a sampling of results (first 6 overlays) in the UI
            sample_overlays = sorted(glob.glob(os.path.join(output_folder, "*_overlay.jpg")))[:6]
            if sample_overlays:
                st.subheader("Sample overlays")
                cols = st.columns(min(3, len(sample_overlays)))
                for idx, p in enumerate(sample_overlays):
                    with cols[idx % 3]:
                        st.image(Image.open(p).convert("RGB"), caption=os.path.basename(p), use_column_width=True)

            st.info(f"Outputs saved to: `{os.path.abspath(output_folder)}`")
