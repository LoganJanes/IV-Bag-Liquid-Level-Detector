import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def detect_iv_bag_level_consolidated(image_path):
    """Consolidated IV bag detection with all processing in 12 steps"""
    
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Can't load image: {image_path}")
        return None
    
    # Initialize variables for analysis
    steps = []
    step_titles = []
    
    # Step 1: Original image
    original = image.copy()
    steps.append(original)
    step_titles.append("1. Original Image")
    
    # Step 2: Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    steps.append(gray_bgr)
    step_titles.append("2. Grayscale Conversion")
    
    # Step 3: Adaptive thresholding for Bag Detection
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    steps.append(binary_bgr)
    step_titles.append("3. Adaptive Thresholding")
    
    # Step 4: Morphological closing to find IV bag
    kernel = np.ones((5, 5), np.uint8)
    morph_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph_closed_bgr = cv2.cvtColor(morph_closed, cv2.COLOR_GRAY2BGR)
    steps.append(morph_closed_bgr)
    step_titles.append("4. Morphological Closing")
    
    # BAG FINDING LOGIC
    contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bag_x = bag_y = bag_w = bag_h = 0
    
    if contours:
        good_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Big enough to be a bag
                x, y, w, h = cv2.boundingRect(contour)
                if w > 0:
                    aspect_ratio = h / w
                    if 1.5 < aspect_ratio < 4.0:  # Bag-like shape
                        good_contours.append((contour, area))
        
        if good_contours:
            biggest_contour = max(good_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(biggest_contour)
            # Add padding
            padding = 5
            height, width = gray.shape
            bag_x = max(0, x - padding)
            bag_y = max(0, y - padding)
            bag_w = min(width - bag_x, w + 2 * padding)
            bag_h = min(height - bag_y, h + 2 * padding)
    
    # Default region if bag detection fails
    if bag_w == 0 or bag_h == 0:
        height, width = gray.shape
        bag_x, bag_y, bag_w, bag_h = width//4, height//6, width//2, 2*height//3
    
    # Step 5: Contour detection + bag highlighting
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.rectangle(contour_image, (bag_x, bag_y), (bag_x + bag_w, bag_y + bag_h), (255, 0, 0), 3)
    steps.append(contour_image)
    step_titles.append("5. Contour Detection + Bag Detection")
    
    # Step 6: Extract bag area
    bag_area = image[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w]
    bag_display = np.zeros_like(image)
    if bag_area.size > 0:
        bag_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = bag_area
    steps.append(bag_display)
    step_titles.append("6. Extracted Bag Area")
    
    # Step 7: HSV conversion of bag area
    hsv_display = np.zeros_like(image)
    if bag_area.size > 0:
        hsv_bag = cv2.cvtColor(bag_area, cv2.COLOR_BGR2HSV)
        hsv_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = hsv_bag
    steps.append(hsv_display)
    step_titles.append("7. HSV Color Space")
    
    # Step 8: Unified liquid mask and Analysis
    liquid_mask_display = np.zeros_like(image)
    color_level = 5.0
    liquid_mask = None
    
    if bag_area.size > 0:
        hsv = cv2.cvtColor(bag_area, cv2.COLOR_BGR2HSV)
        lower_liquid = np.array([0, 30, 80])
        upper_liquid = np.array([180, 255, 255])
        liquid_mask = cv2.inRange(hsv, lower_liquid, upper_liquid)
        
        # COLOR-BASED ANALYSIS
        kernel_small = np.ones((3, 3), np.uint8)
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_OPEN, kernel_small)
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_CLOSE, kernel_small)
        
        # Row-by-row analysis
        height = liquid_mask.shape[0]
        liquid_in_each_row = []
        for row in range(height):
            liquid_pixels = np.sum(liquid_mask[row, :] > 0)
            total_pixels = liquid_mask.shape[1]
            liquid_percentage = liquid_pixels / max(total_pixels, 1)
            liquid_in_each_row.append(liquid_percentage)
            
        threshold = 0.15
        rows_with_liquid = np.array(liquid_in_each_row) > threshold
        
        if np.any(rows_with_liquid):
            liquid_row_numbers = np.where(rows_with_liquid)[0]
            top_liquid_row = liquid_row_numbers[0]
            bottom_liquid_row = liquid_row_numbers[-1]
            liquid_height = bottom_liquid_row - top_liquid_row + 1
            bottom_position = (height - bottom_liquid_row) / height
            liquid_size = liquid_height / height
            
            if liquid_size < 0.2:
                color_level = (1 - bottom_position * 1.2) * 100
                color_level = max(5, min(25, color_level))
            else:
                center_y = (top_liquid_row + bottom_liquid_row) / 2
                center_position = 1 - (center_y / height)
                color_level = (center_position * 0.6 + liquid_size * 0.4) * 100
        
        liquid_mask_bgr = cv2.cvtColor(liquid_mask, cv2.COLOR_GRAY2BGR)
        liquid_mask_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = liquid_mask_bgr
    
    steps.append(liquid_mask_display)
    step_titles.append("8. Unified Liquid Mask + Analysis")
    
    # Step 9: Morphological operations on liquid mask
    morph_mask_display = np.zeros_like(image)
    if bag_area.size > 0 and liquid_mask is not None:
        morph_mask_bgr = cv2.cvtColor(liquid_mask, cv2.COLOR_GRAY2BGR)
        morph_mask_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = morph_mask_bgr
    steps.append(morph_mask_display)
    step_titles.append("9. Final Liquid Mask")
    
    # Step 10: Otsu thresholding  (with Contour Analysis)
    otsu_display = np.zeros_like(image)
    contour_level = 5.0
    
    if bag_area.size > 0:
        bag_gray = cv2.cvtColor(bag_area, cv2.COLOR_BGR2GRAY)
        _, otsu_thresh = cv2.threshold(bag_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours_otsu, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_otsu:
            biggest_contour = max(contours_otsu, key=cv2.contourArea)
            area = cv2.contourArea(biggest_contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(biggest_contour)
                center_y = y + h // 2
                position = 1 - (center_y / bag_h)
                size_factor = min(area / (bag_w * bag_h * 0.5), 1.0)
                contour_level = (position * 0.7 + size_factor * 0.3) * 100
        
        otsu_bgr = cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2BGR)
        otsu_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = otsu_bgr
    
    steps.append(otsu_display)
    step_titles.append("10. Otsu Thresholding + Contour Analysis")
    
    # Step 11: Brightness Analysis and Detection
    brightness_display = image.copy()
    brightness_level = 10.0
    
    if bag_area.size > 0:
        bag_gray = cv2.cvtColor(bag_area, cv2.COLOR_BGR2GRAY)
        height = bag_gray.shape[0]
        num_strips = min(20, height // 3)
        
        if num_strips > 0:
            strip_height = height // num_strips
            brightness_values = []
            
            # BRIGHTNESS ANALYSIS
            for i in range(num_strips):
                start_row = i * strip_height
                end_row = min((i + 1) * strip_height, height)
                strip = bag_gray[start_row:end_row, :]
                if strip.size > 0:
                    avg_brightness = np.mean(strip)
                    brightness_values.append(avg_brightness)
                
                # Draw visualization lines
                y_pos = bag_y + start_row
                cv2.line(brightness_display, (bag_x, y_pos), (bag_x + bag_w, y_pos), (0, 255, 255), 1)
            
            if len(brightness_values) >= 3:
                brightness_values = np.array(brightness_values)
                changes = np.gradient(brightness_values)
                
                # Find significant drops
                big_drops = []
                for i in range(len(changes)):
                    if changes[i] < -10:
                        drop_size = abs(changes[i])
                        big_drops.append((i, drop_size))
                
                if big_drops:
                    biggest_drop_index = max(big_drops, key=lambda x: x[1])[0]
                else:
                    avg_brightness = np.mean(brightness_values)
                    dark_areas = brightness_values < (avg_brightness * 0.8)
                    if np.any(dark_areas):
                        biggest_drop_index = np.where(dark_areas)[0][0]
                    else:
                        biggest_drop_index = len(brightness_values) // 2
                
                brightness_level = (1 - biggest_drop_index / len(brightness_values)) * 100
    
    steps.append(brightness_display)
    step_titles.append("11. Brightness Analysis + Detection")
    
    # Step 12: Final result (CALCULATION AND CLASSIFICATION)

    # Combine all methods
    if color_level <= 30:
        final_level = color_level * 0.8 + brightness_level * 0.2
    else:
        final_level = color_level * 0.6 + brightness_level * 0.3 + contour_level * 0.1
    
    # Special adjustment for very low levels
    if color_level < 15 and brightness_level < 25:
        final_level = min(final_level, 20)
    
    # Clamp to realistic bounds
    final_level = max(2, min(95, final_level))
    
    # Classify into status
    if final_level >= 60:
        status = "high"
    elif final_level >= 30:
        status = "medium"
    else:
        status = "low"
    
    # Create final result visualization
    final_result = image.copy()
    overlay = np.zeros_like(final_result)
    
    if bag_area.size > 0 and liquid_mask is not None:
        # Choose color based on status
        if status == 'low':
            color = (0, 0, 255)  # Red
        elif status == 'medium':
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
            
        # Create liquid overlay
        bag_overlay = overlay[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w]
        bag_overlay[liquid_mask > 0] = color
        transparency = 0.4
        final_result = cv2.addWeighted(final_result, 1-transparency, overlay, transparency, 0)
        
        # Draw contours
        contours_final, _ = cv2.findContours(liquid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_final:
            if cv2.contourArea(contour) > 50:
                adjusted_contour = contour + np.array([bag_x, bag_y])
                cv2.drawContours(final_result, [adjusted_contour], -1, color, 2)
    
    cv2.rectangle(final_result, (bag_x, bag_y), (bag_x + bag_w, bag_y + bag_h), (255, 255, 255), 2)
    steps.append(final_result)
    step_titles.append("12. Final Result + Level Classification")
    
    # Create and save visualization grid
    rows = 3
    cols = 4
    cell_width = 400
    cell_height = 300
    grid_image = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)
    
    for i, (step_img, title) in enumerate(zip(steps, step_titles)):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        resized_step = cv2.resize(step_img, (cell_width - 20, cell_height - 40))
        
        y_start = row * cell_height + 10
        y_end = y_start + cell_height - 40
        x_start = col * cell_width + 10
        x_end = x_start + cell_width - 20
        
        grid_image[y_start:y_end, x_start:x_end] = resized_step
        cv2.putText(grid_image, title, (x_start, y_start - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save results
    filename_without_extension = os.path.splitext(image_path)[0]
    steps_path = f"{filename_without_extension}_processing_steps.jpg"
    result_path = f"{filename_without_extension}_result.jpg"
    
    cv2.imwrite(steps_path, grid_image)
    cv2.imwrite(result_path, final_result)
    
    print(f"Final level: {final_level:.1f}% - Status: {status}")
    return status


if __name__ == "__main__":
    # Find all image files
    image_files = []
    for extension in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(extension))
    
    # Filter out result and processing step images
    original_images = []
    for image_path in image_files:
        if not ('_result' in image_path or '_processing_steps' in image_path):
            original_images.append(image_path)
    
    if original_images:
        print(f"Found {len(original_images)} original images to test:")
        print("-" * 40)
        
        # Test first 5 original images
        for image_path in original_images[:5]:
            try:
                result = detect_iv_bag_level_consolidated(image_path)
                print(f"{image_path}: {result} level")
            except Exception as error:
                print(f"{image_path}: Error - {error}")
    else:
        print("No original image files found in current directory!")