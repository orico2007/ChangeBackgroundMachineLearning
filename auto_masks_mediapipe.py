import os, glob
import cv2
import numpy as np
import mediapipe as mp

IMG_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
os.makedirs(MASK_DIR, exist_ok=True)

THRESH = 0.5
SMOOTH_SIGMA = 5
DILATE_ITERS = 1

def main():
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not files:
        raise RuntimeError(f"No images found in {IMG_DIR}")

    mp_selfie = mp.solutions.selfie_segmentation

    done = 0
    skipped = 0

    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        for ip in files:
            stem = os.path.splitext(os.path.basename(ip))[0]
            outp = os.path.join(MASK_DIR, stem + ".png")

            # ✅ do not overwrite existing masks
            if os.path.exists(outp):
                skipped += 1
                continue

            bgr = cv2.imread(ip)
            if bgr is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = seg.process(rgb)

            if res.segmentation_mask is None:
                print("No mask:", ip)
                continue

            prob = res.segmentation_mask.astype(np.float32)

            if SMOOTH_SIGMA > 0:
                prob = cv2.GaussianBlur(prob, (0, 0), SMOOTH_SIGMA, SMOOTH_SIGMA)

            mask = (prob >= THRESH).astype(np.uint8) * 255

            if DILATE_ITERS > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.dilate(mask, k, iterations=DILATE_ITERS)

            cv2.imwrite(outp, mask)
            done += 1
            print("saved:", outp)

    print(f"Done. created={done} skipped_existing={skipped}")

if __name__ == "__main__":
    main()
