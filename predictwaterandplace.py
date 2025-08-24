import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


b3_path = r"C:\Users\pc\Downloads\EO_Browser_images (1)\2025-08-13-00_00_2025-08-13-23_59_Sentinel-2_L2A_B03_(Raw).tiff"  
b4_path = r"C:\Users\pc\Downloads\EO_Browser_images (1)\2025-08-13-00_00_2025-08-13-23_59_Sentinel-2_L2A_B04_(Raw).tiff"  
b8_path = r"C:\Users\pc\Downloads\EO_Browser_images (1)\2025-08-13-00_00_2025-08-13-23_59_Sentinel-2_L2A_B08_(Raw).tiff"  


try:
    with rasterio.open(b3_path) as g_src, \
         rasterio.open(b4_path) as r_src, \
         rasterio.open(b8_path) as nir_src:

        green = g_src.read(1).astype('float32')
        red = r_src.read(1).astype('float32')
        nir = nir_src.read(1).astype('float32')
        profile = g_src.profile
except Exception as e:
    print(f"Hata: Dosyalar okunamadı veya dosya yolu hatalı. Hata: {e}")
    exit()


ndvi = (nir - red) / (nir + red + 1e-10)
ndwi = (green - nir) / (green + nir + 1e-10)


ndvi_norm = (ndvi - np.nanmin(ndvi)) / (np.nanmax(ndvi) - np.nanmin(ndvi) + 1e-10)
ndwi_clamped = np.clip(ndwi, -1, 1)
ndwi_norm = (ndwi_clamped - np.nanmin(ndwi_clamped)) / (np.nanmax(ndwi_clamped) - np.nanmin(ndwi_clamped) + 1e-10)


rgb = np.zeros((green.shape[0], green.shape[1], 3), dtype=np.float32)

ndvi_gamma = ndvi_norm ** (1/1.5)
ndwi_blue = np.clip(ndwi_norm * 1.5, 0, 1)

rgb[:, :, 0] = 1 - ndvi_gamma
rgb[:, :, 1] = ndvi_gamma
rgb[:, :, 2] = ndwi_blue

for i in range(3):
    rgb[:, :, i] = gaussian_filter(rgb[:, :, i], sigma=1)

rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


profile.update({"count": 3, "dtype": rasterio.uint8})
output_path = r"C:\Users\pc\Downloads\ndvi_ndwi_rgb_heatmap_final_v3.tif"
with rasterio.open(output_path, "w", **profile) as dst:
    
    dst.write(rgb_uint8[:, :, 0], 1)
    dst.write(rgb_uint8[:, :, 1], 2)
    dst.write(rgb_uint8[:, :, 2], 3)

print(f"Geliştirilmiş RGB ısı haritası kaydedildi: {output_path}")


plt.figure(figsize=(12, 12))
plt.imshow(rgb_uint8, interpolation='bilinear')
plt.axis('off')
plt.title("Geliştirilmiş NDVI & NDWI RGB Haritası")
plt.show()