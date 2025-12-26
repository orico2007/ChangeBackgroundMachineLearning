import os, glob, random
import numpy as np
import cv2
import tensorflow as tf

IMG_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
OUT_DIR = "runs_tf"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "unet_bgblur.keras")

IMG_SIZE = 256
BATCH = 8
EPOCHS_MORE = 1000     # train 10 more epochs each run
VAL_SPLIT = 0.15
SEED = 42
LR = 1e-3

def list_pairs():
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    pairs = []
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(MASK_DIR, stem + ".png")
        if os.path.exists(mp):
            pairs.append((ip, mp))
    if len(pairs) < 10:
        raise RuntimeError("Need at least 10 image/mask pairs.")
    return pairs

def read_img_mask(ip, mp):
    img = cv2.cvtColor(cv2.imread(ip.decode()), cv2.COLOR_BGR2RGB)
    m = cv2.imread(mp.decode(), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    m = (m > 127).astype(np.float32)[..., None]
    return img, m

def tf_read(ip, mp):
    img, m = tf.numpy_function(read_img_mask, [ip, mp], [tf.float32, tf.float32])
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    m.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img, m

def augment(img, m):
    if tf.random.uniform(()) < 0.5:
        img = tf.image.flip_left_right(img)
        m = tf.image.flip_left_right(m)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, m

def conv_block(x, f):
    x = tf.keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def build_unet():
    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    c1 = conv_block(inp, 32); p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_block(p1, 64);  p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_block(p2, 128); p3 = tf.keras.layers.MaxPool2D()(c3)
    c4 = conv_block(p3, 256); p4 = tf.keras.layers.MaxPool2D()(c4)
    b  = conv_block(p4, 512)
    u4 = tf.keras.layers.UpSampling2D()(b);  u4 = tf.keras.layers.Concatenate()([u4, c4]); c5 = conv_block(u4, 256)
    u3 = tf.keras.layers.UpSampling2D()(c5); u3 = tf.keras.layers.Concatenate()([u3, c3]); c6 = conv_block(u3, 128)
    u2 = tf.keras.layers.UpSampling2D()(c6); u2 = tf.keras.layers.Concatenate()([u2, c2]); c7 = conv_block(u2, 64)
    u1 = tf.keras.layers.UpSampling2D()(c7); u1 = tf.keras.layers.Concatenate()([u1, c1]); c8 = conv_block(u1, 32)
    out = tf.keras.layers.Conv2D(1, 1, activation=None)(c8)  # logits
    return tf.keras.Model(inp, out)

def dice_loss(y_true, logits, eps=1e-6):
    y_prob = tf.nn.sigmoid(logits)
    num = 2.0 * tf.reduce_sum(y_prob * y_true, axis=[1,2,3]) + eps
    den = tf.reduce_sum(y_prob + y_true, axis=[1,2,3]) + eps
    return 1.0 - tf.reduce_mean(num / den)

def loss_fn(y_true, logits):
    bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits))
    return bce + dice_loss(y_true, logits)

def iou_metric(y_true, logits, thr=0.5, eps=1e-6):
    y_prob = tf.nn.sigmoid(logits)
    y_pred = tf.cast(y_prob > thr, tf.float32)
    inter = tf.reduce_sum(y_pred * y_true, axis=[1,2,3])
    union = tf.reduce_sum(y_pred + y_true - y_pred*y_true, axis=[1,2,3])
    return tf.reduce_mean((inter + eps) / (union + eps))

def main():
    random.seed(SEED)

    pairs = list_pairs()
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_SPLIT))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_ips = [p[0] for p in train_pairs]
    train_mps = [p[1] for p in train_pairs]
    val_ips = [p[0] for p in val_pairs]
    val_mps = [p[1] for p in val_pairs]

    train_ds = tf.data.Dataset.from_tensor_slices((train_ips, train_mps))
    train_ds = train_ds.shuffle(512, seed=SEED).map(tf_read, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_ips, val_mps))
    val_ds = val_ds.map(tf_read, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    # ✅ Resume if exists
    if os.path.exists(MODEL_PATH):
        print("Resuming from:", MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    else:
        print("No existing model found. Building new.")
        model = build_unet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=loss_fn,
        metrics=[iou_metric]
    )

    cb = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_iou_metric",
            mode="max",
            save_best_only=True
        )
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_MORE, callbacks=cb)
    print("Done. Best saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
