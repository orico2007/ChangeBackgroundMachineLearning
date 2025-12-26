import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "runs_tf/unet_bgblur.keras"

IMG_SIZE = 256
BG_BLUR_SIGMA = 18
MASK_EDGE_SIGMA = 6
TEMP_SMOOTH = 0.85

def main():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={},  # we only need inference
        compile=False
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    prev_mask = None
    print("q=quit")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        x = small[None, ...]  # 1,H,W,3
        logits = model(x, training=False).numpy()[0, :, :, 0]
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))  # sigmoid

        mask = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=MASK_EDGE_SIGMA, sigmaY=MASK_EDGE_SIGMA)

        if prev_mask is None:
            prev_mask = mask
        mask = TEMP_SMOOTH * prev_mask + (1 - TEMP_SMOOTH) * mask
        prev_mask = mask
        mask = np.clip(mask, 0.0, 1.0)

        blurred = cv2.GaussianBlur(frame_bgr, (0,0), sigmaX=BG_BLUR_SIGMA, sigmaY=BG_BLUR_SIGMA)
        mask3 = np.repeat(mask[:, :, None], 3, axis=2)
        out = (mask3 * frame_bgr + (1 - mask3) * blurred).astype(np.uint8)

        cv2.imshow("TF U-Net background blur (q=quit)", out)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
