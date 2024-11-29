    # Get centroid
    cx, cy = centroids[i]
    cx, cy = int(cx), int(cy)
    
    # Get bounding box stats
    max_horizontal_distance  = stats[i, cv2.CC_STAT_WIDTH]
    max_vertical_distance = stats[i, cv2.CC_STAT_HEIGHT]

    cur_d = min(max_horizontal_distance, max_vertical_distance)
    
    if cur_d >= d_25_min and cur_d <= d_25_max:
      valid_coins.append((cx, cy, 25))

    elif cur_d >= d_5_min and cur_d <= d_5_max:
      valid_coins.append((cx, cy, 5))

    elif cur_d >= d_1_min and cur_d <= d_1_max:
      valid_coins.append((cx, cy, 1))