import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def find_iv_bag(image):
    """Find the IV bag in the image"""
    # Step 1: Convert the image to grayscale to remove color information and simplify processing. We do this to find the boundaries and use the colour later to detect the actual liquid.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply adaptive thresholding to create a binary image that separates objects from background
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    
    # Step 3: Apply morphological closing to fill small gaps and connect broken contours
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step 4: Find contours to detect object boundaries in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        good_contours = []
        for contour in contours:
            # Step 5: Filter contours by area to find objects large enough to be IV bags
            area = cv2.contourArea(contour)
            if area > 5000:  # Big enough to be a bag
                # Step 6: Calculate bounding rectangle and aspect ratio to identify bag-like shapes
                x, y, w, h = cv2.boundingRect(contour)
                if w > 0:
                    aspect_ratio = h / w
                    if 1.5 < aspect_ratio < 4.0:  # Bag-like shape
                        good_contours.append((contour, area))
        
        if good_contours:
            # Step 7: Select the largest qualifying contour as the IV bag
            biggest_contour = max(good_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(biggest_contour)
            
            # Step 8: Add padding around the detected bag area for better processing
            padding = 5
            height, width = gray.shape
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            return x, y, w, h
    
    # Step 9: If bag detection fails, return a default region in the center of the image
    height, width = gray.shape
    return width//4, height//6, width//2, 2*height//3


def find_liquid_by_color(image, bag_coords):
    """Look for liquid using colors"""
    x, y, w, h = bag_coords
    bag_area = image[y:y+h, x:x+w]
    
    if bag_area.size == 0:
        return 5.0, None
    
    # Step 1: Convert BGR color space to HSV for more robust color detection
    hsv = cv2.cvtColor(bag_area, cv2.COLOR_BGR2HSV)

    # Look for different liquid colors
    all_masks = []

    # Step 2: Create color masks for red liquid using HSV color ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # Step 3: Combine red masks using bitwise OR operation since red wraps around in HSV
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    all_masks.append(red_mask)

    # Step 4: Create color mask for blue liquid using HSV color range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    all_masks.append(blue_mask)

    # Step 5: Create color mask for clear/transparent liquid using broader HSV range
    lower_clear = np.array([0, 30, 80])
    upper_clear = np.array([180, 255, 255])
    clear_mask = cv2.inRange(hsv, lower_clear, upper_clear)
    all_masks.append(clear_mask)

    # Step 6: Combine all color masks using bitwise OR to detect any liquid color
    final_mask = np.zeros_like(all_masks[0])
    for mask in all_masks:
        final_mask = cv2.bitwise_or(final_mask, mask)

    # Step 7: Apply morphological opening to remove noise from the combined mask
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    # Step 8: Apply morphological closing to fill small gaps in the detected liquid regions
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # Step 9: Analyze each horizontal row to determine liquid presence and distribution
    height = final_mask.shape[0]
    liquid_in_each_row = []

    for row in range(height):
        # Step 10: Calculate the percentage of liquid pixels in each row
        liquid_pixels = np.sum(final_mask[row, :] > 0)
        total_pixels = final_mask.shape[1]
        liquid_percentage = liquid_pixels / max(total_pixels, 1)
        liquid_in_each_row.append(liquid_percentage)

    # Step 11: Apply threshold to determine which rows contain significant liquid
    threshold = 0.15
    rows_with_liquid = np.array(liquid_in_each_row) > threshold

    if np.any(rows_with_liquid):
        # Step 12: Find the top and bottom boundaries of the liquid region
        liquid_row_numbers = np.where(rows_with_liquid)[0]
        top_liquid_row = liquid_row_numbers[0]
        bottom_liquid_row = liquid_row_numbers[-1]

        # Step 13: Calculate liquid height and position metrics
        liquid_height = bottom_liquid_row - top_liquid_row + 1
        bottom_position = (height - bottom_liquid_row) / height
        liquid_size = liquid_height / height

        # Step 14: Calculate liquid level percentage based on size and position
        if liquid_size < 0.2:
            level = (1 - bottom_position * 1.2) * 100
            level = max(5, min(25, level))
        else:
            center_y = (top_liquid_row + bottom_liquid_row) / 2
            center_position = 1 - (center_y / height)
            level = (center_position * 0.6 + liquid_size * 0.4) * 100

        return level, final_mask
    else:
        return 5.0, final_mask


def check_brightness_changes(bag_area):
    """Look for brightness changes to find liquid level"""
    if bag_area.size == 0:
        return 10.0
    
    height, width = bag_area.shape
    num_strips = min(20, height // 3)
    if num_strips <= 0:
        return 10.0
    
    strip_height = height // num_strips
    brightness_values = []

    # Step 1: Divide the image into horizontal strips for brightness analysis
    for i in range(num_strips):
        start_row = i * strip_height
        end_row = min((i + 1) * strip_height, height)
        strip = bag_area[start_row:end_row, :]
        if strip.size > 0:
            # Step 2: Calculate average brightness for each horizontal strip
            avg_brightness = np.mean(strip)
            brightness_values.append(avg_brightness)

    if len(brightness_values) < 3:
        return 10.0

    # Step 3: Calculate gradient to detect sudden brightness changes between strips
    brightness_values = np.array(brightness_values)
    changes = np.gradient(brightness_values)

    # Step 4: Identify significant brightness drops that indicate liquid boundaries
    big_drops = []
    for i in range(len(changes)):
        if changes[i] < -10:
            drop_size = abs(changes[i])
            big_drops.append((i, drop_size))

    if big_drops:
        # Step 5: Find the most significant brightness drop as liquid boundary indicator
        biggest_drop_index = max(big_drops, key=lambda x: x[1])[0]
    else:
        # Step 6: If no major drops, look for areas significantly darker than average
        avg_brightness = np.mean(brightness_values)
        dark_areas = brightness_values < (avg_brightness * 0.8)
        if np.any(dark_areas):
            biggest_drop_index = np.where(dark_areas)[0][0]
        else:
            biggest_drop_index = len(brightness_values) // 2

    # Step 7: Convert the detected boundary position to a liquid level percentage
    level = (1 - biggest_drop_index / len(brightness_values)) * 100
    return level


def detect_iv_bag_level(image_path):
    """Main function to detect IV bag liquid level"""
    # Step 1: Load the input image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Can't load image: {image_path}")
        return None
    
    # Step 2: Detect the IV bag location within the image
    bag_x, bag_y, bag_w, bag_h = find_iv_bag(image)
    
    # Step 3: Convert the image to grayscale for brightness-based analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bag_area = gray[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w]

    # Step 4: Apply color-based liquid detection method
    color_level, liquid_mask = find_liquid_by_color(image, (bag_x, bag_y, bag_w, bag_h))
    # Step 5: Apply brightness-based liquid detection method
    brightness_level = check_brightness_changes(bag_area)

    # Step 6: Apply Otsu's thresholding for automatic binary threshold selection
    _, binary = cv2.threshold(bag_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Step 7: Find contours in the thresholded image for shape-based analysis
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_level = 5.0
    if contours:
        # Step 8: Select the largest contour as the main liquid region
        biggest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest_contour)
        if area > 100:
            # Step 9: Calculate bounding rectangle and position metrics from contour
            x, y, w, h = cv2.boundingRect(biggest_contour)
            center_y = y + h // 2
            position = 1 - (center_y / bag_h)
            size_factor = min(area / (bag_w * bag_h * 0.5), 1.0)
            # Step 10: Combine position and size factors for contour-based level estimation
            contour_level = (position * 0.7 + size_factor * 0.3) * 100

    # Step 11: Combine results from all detection methods using weighted averaging
    if color_level <= 30:
        final_level = color_level * 0.8 + brightness_level * 0.2
    else:
        final_level = color_level * 0.6 + brightness_level * 0.3 + contour_level * 0.1

    # Step 12: Apply special adjustment for very low liquid levels
    if color_level < 15 and brightness_level < 25:
        final_level = min(final_level, 20)

    # Step 13: Clamp the final level to realistic bounds
    final_level = max(2, min(95, final_level))

    # Step 14: Classify liquid level into status categories
    if final_level >= 60:
        status = "high"
    elif final_level >= 30:
        status = "medium"
    else:
        status = "low"

    # Step 15: Save visualization image showing the detection results
    save_result_image(image, (bag_x, bag_y, bag_w, bag_h), status, liquid_mask, image_path)

    return status


def save_result_image(image, bag_coords, status, liquid_mask, original_path):
    """Save an image showing the detected liquid"""
    bag_x, bag_y, bag_w, bag_h = bag_coords
    result_image = image.copy()
    overlay = np.zeros_like(result_image)

    if liquid_mask is not None and liquid_mask.size > 0:
        # Step 1: Choose visualization color based on detected liquid status
        if status == 'low':
            color = (0, 0, 255)  # Red
        elif status == 'medium':
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green

        # Step 2: Create colored overlay for detected liquid regions
        bag_overlay = overlay[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w]
        bag_overlay[liquid_mask > 0] = color
        # Step 3: Blend the overlay with the original image using alpha blending
        transparency = 0.4
        result_image = cv2.addWeighted(result_image, 1-transparency, overlay, transparency, 0)

        # Step 4: Find contours in the liquid mask for boundary visualization
        contours, _ = cv2.findContours(liquid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                # Step 5: Adjust contour coordinates to match the original image coordinate system
                adjusted_contour = contour + np.array([bag_x, bag_y])
                # Step 6: Draw contour boundaries on the result image
                cv2.drawContours(result_image, [adjusted_contour], -1, color, 2)

    # Step 7: Draw bounding rectangle around the detected bag area
    cv2.rectangle(result_image, (bag_x, bag_y), (bag_x + bag_w, bag_y + bag_h), (255, 255, 255), 1)

    # Step 8: Save the annotated result image to disk
    filename_without_extension = os.path.splitext(original_path)[0]
    result_path = f"{filename_without_extension}_result.jpg"
    cv2.imwrite(result_path, result_image)

    return result_path


def save_processing_steps_visualization(image, bag_coords, status, original_path):
    """Create a comprehensive visualization showing all image processing steps"""
    bag_x, bag_y, bag_w, bag_h = bag_coords
    
    # Recreate all processing steps for visualization
    steps = []
    step_titles = []
    
    # Step 1: Original image
    original = image.copy()
    steps.append(original)
    step_titles.append("1. Original Image")
    
    # Step 2: Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for display
    steps.append(gray_bgr)
    step_titles.append("2. Grayscale Conversion")
    
    # Step 3: Adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    steps.append(binary_bgr)
    step_titles.append("3. Adaptive Thresholding")
    
    # Step 4: Morphological closing
    kernel = np.ones((5, 5), np.uint8)
    morph_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph_closed_bgr = cv2.cvtColor(morph_closed, cv2.COLOR_GRAY2BGR)
    steps.append(morph_closed_bgr)
    step_titles.append("4. Morphological Closing")
    
    # Step 5: Contour detection for bag finding
    contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.rectangle(contour_image, (bag_x, bag_y), (bag_x + bag_w, bag_y + bag_h), (255, 0, 0), 3)
    steps.append(contour_image)
    step_titles.append("5. Contour Detection + Bag Detection")
    
    # Step 6: Extract bag area
    bag_area = image[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w]
    # Resize to match other images for display
    bag_display = np.zeros_like(image)
    if bag_area.size > 0:
        bag_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = bag_area
    steps.append(bag_display)
    step_titles.append("6. Extracted Bag Area")
    
    # Step 7: HSV conversion of bag area
    if bag_area.size > 0:
        hsv_bag = cv2.cvtColor(bag_area, cv2.COLOR_BGR2HSV)
        hsv_display = np.zeros_like(image)
        hsv_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = hsv_bag
    else:
        hsv_display = np.zeros_like(image)
    steps.append(hsv_display)
    step_titles.append("7. HSV Color Space")
    
    # Step 8: Red liquid mask
    red_mask_display = np.zeros_like(image)
    if bag_area.size > 0:
        hsv = cv2.cvtColor(bag_area, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        red_mask_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = red_mask_bgr
    steps.append(red_mask_display)
    step_titles.append("8. Red Liquid Mask")
    
    # Step 9: Blue liquid mask
    blue_mask_display = np.zeros_like(image)
    if bag_area.size > 0:
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_mask_bgr = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
        blue_mask_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = blue_mask_bgr
    steps.append(blue_mask_display)
    step_titles.append("9. Blue Liquid Mask")
    
    # Step 10: Clear liquid mask
    clear_mask_display = np.zeros_like(image)
    if bag_area.size > 0:
        lower_clear = np.array([0, 30, 80])
        upper_clear = np.array([180, 255, 255])
        clear_mask = cv2.inRange(hsv, lower_clear, upper_clear)
        clear_mask_bgr = cv2.cvtColor(clear_mask, cv2.COLOR_GRAY2BGR)
        clear_mask_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = clear_mask_bgr
    steps.append(clear_mask_display)
    step_titles.append("10. Clear Liquid Mask")
    
    # Step 11: Combined color masks
    combined_mask_display = np.zeros_like(image)
    if bag_area.size > 0:
        combined_mask = cv2.bitwise_or(red_mask, blue_mask)
        combined_mask = cv2.bitwise_or(combined_mask, clear_mask)
        combined_mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        combined_mask_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = combined_mask_bgr
    steps.append(combined_mask_display)
    step_titles.append("11. Combined Color Masks")
    
    # Step 12: Morphological operations on combined mask
    morph_mask_display = np.zeros_like(image)
    if bag_area.size > 0:
        kernel_small = np.ones((3, 3), np.uint8)
        morph_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        morph_final = cv2.morphologyEx(morph_opened, cv2.MORPH_CLOSE, kernel_small)
        morph_mask_bgr = cv2.cvtColor(morph_final, cv2.COLOR_GRAY2BGR)
        morph_mask_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = morph_mask_bgr
    steps.append(morph_mask_display)
    step_titles.append("12. Morphology on Color Mask")
    
    # Step 13: Otsu thresholding
    otsu_display = np.zeros_like(image)
    if bag_area.size > 0:
        bag_gray = cv2.cvtColor(bag_area, cv2.COLOR_BGR2GRAY)
        _, otsu_thresh = cv2.threshold(bag_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_bgr = cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2BGR)
        otsu_display[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w] = otsu_bgr
    steps.append(otsu_display)
    step_titles.append("13. Otsu Thresholding")
    
    # Step 14: Brightness analysis visualization
    brightness_display = image.copy()
    if bag_area.size > 0:
        bag_gray = cv2.cvtColor(bag_area, cv2.COLOR_BGR2GRAY)
        height = bag_gray.shape[0]
        num_strips = min(20, height // 3)
        if num_strips > 0:
            strip_height = height // num_strips
            for i in range(num_strips):
                start_row = i * strip_height
                end_row = min((i + 1) * strip_height, height)
                # Draw horizontal lines to show brightness analysis strips
                y_pos = bag_y + start_row
                cv2.line(brightness_display, (bag_x, y_pos), (bag_x + bag_w, y_pos), (0, 255, 255), 1)
    steps.append(brightness_display)
    step_titles.append("14. Brightness Analysis Strips")
    
    # Step 15: Final result with liquid detection
    final_result = image.copy()
    overlay = np.zeros_like(final_result)
    
    if bag_area.size > 0:
        # Choose color based on status
        if status == 'low':
            color = (0, 0, 255)  # Red
        elif status == 'medium':
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
            
        # Create the final liquid overlay
        if 'morph_final' in locals():
            bag_overlay = overlay[bag_y:bag_y+bag_h, bag_x:bag_x+bag_w]
            bag_overlay[morph_final > 0] = color
            transparency = 0.4
            final_result = cv2.addWeighted(final_result, 1-transparency, overlay, transparency, 0)
            
            # Draw contours
            contours_final, _ = cv2.findContours(morph_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_final:
                if cv2.contourArea(contour) > 50:
                    adjusted_contour = contour + np.array([bag_x, bag_y])
                    cv2.drawContours(final_result, [adjusted_contour], -1, color, 2)
    
    cv2.rectangle(final_result, (bag_x, bag_y), (bag_x + bag_w, bag_y + bag_h), (255, 255, 255), 2)
    steps.append(final_result)
    step_titles.append("15. Final Result with Detection")
    
    # Create a grid layout to show all steps
    rows = 4
    cols = 4
    img_height, img_width = image.shape[:2]
    
    # Resize images for grid display
    cell_width = 400
    cell_height = 300
    
    grid_image = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)
    
    for i, (step_img, title) in enumerate(zip(steps, step_titles)):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        # Resize the step image
        resized_step = cv2.resize(step_img, (cell_width - 20, cell_height - 40))
        
        # Calculate position in grid
        y_start = row * cell_height + 10
        y_end = y_start + cell_height - 40
        x_start = col * cell_width + 10
        x_end = x_start + cell_width - 20
        
        # Place the image
        grid_image[y_start:y_end, x_start:x_end] = resized_step
        
        # Add title text
        cv2.putText(grid_image, title, (x_start, y_start - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save the comprehensive processing steps visualization
    filename_without_extension = os.path.splitext(original_path)[0]
    steps_path = f"{filename_without_extension}_processing_steps.jpg"
    cv2.imwrite(steps_path, grid_image)
    
    return steps_path


# Test the code
if __name__ == "__main__":
    # Find all image files
    image_files = []
    for extension in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(extension))
    
    if image_files:
        print(f"Found {len(image_files)} images to test:")
        print("-" * 40)
        
        # Test first 5 images
        for image_path in image_files[:5]:
            try:
                result = detect_iv_bag_level(image_path)
                print(f"{image_path}: {result} level")
            except Exception as error:
                print(f"{image_path}: Error - {error}")
    else:
        print("No image files found in current directory!")

    print("\nHow to use this code:")
    print('result = detect_iv_bag_level("my_iv_bag_photo.jpg")')
    print('print(result)  # Will show: "high", "medium", or "low"')
    print("# Also creates a result image showing the detected liquid")