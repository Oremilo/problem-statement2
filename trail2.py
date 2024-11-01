
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

raw_image_path = r"D:\eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"

try:
    # Directly load as binary if rawpy fails
    image = np.fromfile(raw_image_path, dtype=np.uint16).reshape((1280, 1920))  # Match your image dimensions
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print("Loaded RAW image successfully.")
except Exception as e:
    print(f"Error loading RAW image: {e}")
else:
    # Proceed with the remaining processing steps
    plt.imshow(image, cmap='gray')
    plt.title("Input Image")
    plt.axis("off")
    plt.show()
    # Continue with denoising and other processes as in your original code...

    # Denoising Techniques
    median_denoised = cv2.medianBlur(image, 5)
    bilateral_denoised = cv2.bilateralFilter(image, 9, 75, 75)
    gaussian_denoised = cv2.GaussianBlur(image, (5, 5), 0)

    # Display Denoised Images
    for denoised, title in zip([median_denoised, bilateral_denoised, gaussian_denoised], 
                                ["Median Filter Denoised", "Bilateral Filter Denoised", "Gaussian Filter Denoised"]):
        plt.imshow(denoised, cmap='gray')
        plt.title(title)
        plt.axis("off")
        plt.show()

    # AI-based Denoising
    try:
        model = tf.keras.models.load_model('path_to_pretrained_model')
        input_image = np.expand_dims(image, axis=(0, -1))  # Adjust dimensions for model input
        denoised_image = model.predict(input_image)
        plt.imshow(np.squeeze(denoised_image), cmap='gray')
        plt.title("AI Denoised Image")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error loading or processing with AI model: {e}")

    # Edge Enhancement with Laplacian
    # Increase contrast of the original image before applying Laplacian
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    high_contrast_image = clahe.apply(image)

    # Apply Laplacian and normalize for better visualization
    laplacian_enhanced = cv2.Laplacian(high_contrast_image, cv2.CV_64F)
    laplacian_enhanced = cv2.normalize(laplacian_enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    plt.imshow(laplacian_enhanced, cmap='gray')
    plt.title("Laplacian Edge Enhancement")
    plt.axis("off")
    plt.show()

    # Compute SNR
    def compute_snr(original, processed):
        signal = np.mean(original)
        noise = np.mean(np.abs(original - processed))
        snr = 20 * np.log10(signal / noise)
        return snr

    snr_median = compute_snr(image, median_denoised)
    snr_bilateral = compute_snr(image, bilateral_denoised)
    snr_gaussian = compute_snr(image, gaussian_denoised)

    print(f"SNR for Median Filter: {snr_median:.2f} dB")
    print(f"SNR for Bilateral Filter: {snr_bilateral:.2f} dB")
    print(f"SNR for Gaussian Filter: {snr_gaussian:.2f} dB")

    # Save Results
    cv2.imwrite('output_median.png', median_denoised)
    cv2.imwrite('output_bilateral.png', bilateral_denoised)
    cv2.imwrite('output_gaussian.png', gaussian_denoised)

    # Summary Report
    print(f"""
    | Method      | SNR (dB)           |
    |-------------|---------------------|
    | Median      | {snr_median:.2f}       |
    | Bilateral   | {snr_bilateral:.2f}       |
    | Gaussian    | {snr_gaussian:.2f}       |
    """)
