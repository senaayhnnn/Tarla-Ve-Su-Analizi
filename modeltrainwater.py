import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# ==== 1. Veri Yollarını Tanımla ====
# Dosyaları düzenlediğiniz ana klasörün yolunu buraya yazın.
dataset_path = r"C:\Users\pc\Downloads\archive"
train_images_path = os.path.join(dataset_path, "train", "images")
train_masks_path = os.path.join(dataset_path, "train", "masks")
test_images_path = os.path.join(dataset_path, "test", "images")
test_masks_path = os.path.join(dataset_path, "test", "masks")
valid_images_path = os.path.join(dataset_path, "valid", "images")
valid_masks_path = os.path.join(dataset_path, "valid", "masks")

IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_CLASSES = 7  # DeepGlobe veri setinde 7 sınıf vardır.

# ==== Sınıf Eşleştirme Sözlüğü Oluşturma ====
try:
    class_dict_path = os.path.join(dataset_path, "class_dict.csv")
    class_df = pd.read_csv(class_dict_path)
    
    color_map = {}
    for index, row in class_df.iterrows():
        color_tuple = (row['r'], row['g'], row['b'])
        color_map[color_tuple] = index
except Exception as e:
    print(f"Hata: class_dict.csv dosyası okunamadı. Hata: {e}")
    exit()

# ==== 2. Veri Setini Yükleme ve Ön İşleme ====
def load_and_preprocess_data(image_path, mask_path):
    try:
        # Görüntüleri yükle
        image = Image.open(image_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
        mask = Image.open(mask_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))

        image_np = np.array(image) / 255.0
        mask_np = np.array(mask)

        # Maskenin RGB değerlerini sınıf ID'lerine dönüştür
        segmentation_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.int32)
        for i in range(IMG_HEIGHT):
            for j in range(IMG_WIDTH):
                pixel_color = tuple(mask_np[i, j])
                segmentation_mask[i, j] = color_map.get(pixel_color, 0)
        
        # Sınıf ID'lerini one-hot encoding'e dönüştür
        mask_one_hot = keras.utils.to_categorical(segmentation_mask, num_classes=NUM_CLASSES)
        return image_np, mask_one_hot
    except Exception as e:
        print(f"Hata: Veri yüklenemedi. Dosya yolu veya formatı kontrol edin. Hata: {e}")
        return None, None

def create_dataset(images_dir, masks_dir):
    try:
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))])
        mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])
        
        if not image_files or not mask_files:
            print(f"Hata: Klasörlerde dosya bulunamadı. Yollar: {images_dir} ve {masks_dir}")
            return np.array([]), np.array([])
            
        images = []
        masks = []
        
        for img_file, mask_file in zip(image_files, mask_files):
            # Dosya isimlerinin eşleştiğinden emin olun
            img_name_without_ext = os.path.splitext(img_file)[0]
            mask_name_without_ext = os.path.splitext(mask_file)[0]
            if img_name_without_ext == mask_name_without_ext.replace('_mask', '_sat'):
                img, mask = load_and_preprocess_data(os.path.join(images_dir, img_file), os.path.join(masks_dir, mask_file))
                if img is not None and mask is not None:
                    images.append(img)
                    masks.append(mask)

        return np.array(images), np.array(masks)
    except FileNotFoundError:
        print("Hata: Belirtilen klasör yolları bulunamadı. Lütfen dosya yollarınızı kontrol edin.")
        return np.array([]), np.array([])
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
        return np.array([]), np.array([])

print("Eğitim verileri yükleniyor...")
train_images, train_masks = create_dataset(train_images_path, train_masks_path)
print(f"Yüklenen Eğitim Görüntüsü Sayısı: {len(train_images)}")

if len(train_images) == 0:
    print("Veri seti boş. Lütfen klasörlerin doğru olduğundan emin olun.")
    exit()

# ==== 3. U-Net Modeli Oluşturma ve Eğitme ====
def build_unet(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), num_classes=7):
    inputs = keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    merge4 = layers.concatenate([conv2, up4], axis=3)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge4)
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    merge5 = layers.concatenate([conv1, up5], axis=3)
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge5)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv5)
    model = keras.Model(inputs, outputs)
    return model

model = build_unet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Modeli eğit
model.fit(train_images, train_masks, epochs=50, batch_size=8, validation_split=0.2)

# Modeli kaydet
model_path = r"C:\Users\pc\Downloads\segmentation_model.keras"
model.save(model_path)
print(f"\nModel kaydedildi: {model_path}")