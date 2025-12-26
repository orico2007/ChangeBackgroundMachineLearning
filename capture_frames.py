import cv2, os, re

OUT_DIR = "dataset/images"
os.makedirs(OUT_DIR, exist_ok=True)

def next_index(folder):
    pat = re.compile(r"^(\d{6})\.jpg$", re.IGNORECASE)
    mx = -1
    for f in os.listdir(folder):
        m = pat.match(f)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    i = next_index(OUT_DIR)
    print(f"s=save, q=quit | starting at {i:06d}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # if your OpenCV GUI works:
        cv2.imshow("capture", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("s"):
            path = os.path.join(OUT_DIR, f"{i:06d}.jpg")
            cv2.imwrite(path, frame)
            print("saved", path)
            i += 1
        elif k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
