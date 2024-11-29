'''

Name: Liza Jivnani
U# : U16181370

This program takes an image as an input and detects the number and positions of the coins of following denominations:
1.) 25 cents 2.) 5 cents 3.) 10 cents 4.) 1 cent

The program leverages the fusion of 2 main proceses: 1.) Segmentation 2) Edge detection. 
Each image is first processed using segmentation. If segmentation detects too little or too many coins, attempt edge detection.

Finally, tthe method with maximum detection rate is chosen to display the final output.

'''

# import all the relevant libraries

import cv2
import numpy as np




#--------------------------------------- Approach 1: Segmentation ------------------------------------


# ----------- 1.1) Define Coin Dimensions and other constants for segmentation ----------


# resize the image using the following downsizing factor. Same for both approaches.
downsize_factor = 7

# 25 cent min and max thresholds 
d_25_min = 14
d_25_max = 17

# 5 cent min and max thresholds 
d_5_min = 10
d_5_max = 11

# 1 cent min and max thresholds 
d_1_min = 7
d_1_max = 8

# 10 cent min and max thresholds 
d_10_min = 5
d_10_max = 6


# ------------ 1.2) Read and Resize the image ---------------


file_name = input()
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

new_width = len(img[0])//downsize_factor
new_height = len(img)//downsize_factor
img = cv2.resize(img, (new_width, new_height))


#----------- 1.3) Render the binary image using the 20 margin pixels which is guranteed to have no coins ------

# use this function to calculate the distances between the mean background color and each point in the image
def dist(p1, p2):
  return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)/np.sqrt(3)


# 1.3.1) Read the image in color
img_color = cv2.imread(file_name, cv2.IMREAD_COLOR)/255.
height, width = img_color.shape[:2]
img_color = cv2.resize(img_color, (width//downsize_factor, height//downsize_factor))
height, width = img_color.shape[:2]

# 1.3.2) Calculate the mean background color using the top, botton, left and right margins of size 20 px
top_mean = cv2.mean(img_color[:20, :])[:3]
bottom_mean = cv2.mean(img_color[-20:, :])[:3]
left_mean = cv2.mean(img_color[:, :20])[:3]
right_mean = cv2.mean(img_color[:, -20:])[:3]

bg_color = np.mean([top_mean, bottom_mean, left_mean, right_mean], axis=0)


# 1.3.3) Store the between each point in the image and the mean background color using the dist function
diff_image = np.zeros((height, width), dtype=np.float64)
for i in range(height):
  for j in range(width):
    diff_image[i,j] = dist(img_color[i,j], bg_color)


# 1.3.4) Compute the bunary image, where all the points above the threshold are 1 (reprecenting white) and the rest (background pixels) are 0 (representing black).
isBinary_threshold = 0.2

binary = np.zeros((height, width), dtype=np.uint8)
for i in range(height):
  for j in range(width):
    if diff_image[i,j] > isBinary_threshold:
      binary[i,j] = 1
    else:
      binary[i,j] = 0


#------------------- 1.4) morph and find components -------------------

# 1.4.1)Morph the image to remove the touching coins and noise from the coin texture
kernel_size = 3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
morphed = binary

morphed = cv2.dilate(morphed, kernel, iterations=2)
morphed = cv2.erode(morphed, kernel, iterations=12)

# 1.4.2) Find connected components in the morphed image so that each component is a coin
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed)


# -------------------1.5) Find Valid Coins -------------------------------------

valid_coins = list()

# iterate through all the components (skip background, component 0).
# Compute the max horizontal and vertical distance of the smallest bounding box that fits the whole component
# Compare the min of these 2 distances with the coin dimensions and if they match store the centrod if the component along with the coin value in valid_coins.
# min of vertical and horizantal distance is used to avoid noise and increase the odds of detecting a coin. 
for i in range(1, num_labels):

    # Get centroid
    cx, cy = centroids[i]
    cx, cy = int(cx), int(cy)
    
    # Get bounding box stats
    max_horizontal_distance  = stats[i, cv2.CC_STAT_WIDTH]
    max_vertical_distance = stats[i, cv2.CC_STAT_HEIGHT]

    cur_d = min(max_horizontal_distance,  max_vertical_distance)
    
    if cur_d >= d_25_min and cur_d <= d_25_max:
      valid_coins.append((cx, cy, 25))

    elif cur_d >= d_5_min and cur_d <= d_5_max:
      valid_coins.append((cx, cy, 5))

    elif cur_d >= d_1_min and cur_d <= d_1_max:
      valid_coins.append((cx, cy, 1))

    elif cur_d >= d_10_min and cur_d <= d_10_max:
      valid_coins.append((cx, cy, 10))



#-------------------------------------------------- Approach 2: Edge Detection -------------------------------------------------------

# if segmentation didn't work, detected too little or too many coins
if len(valid_coins) <= 4 or num_labels > 90:

  # ---------- 2.1) Define Coin Dimensions and other constants for edge detection ------------------
  # These thresholds are defined as the squares of the diameters as these they are used to compute the eucledian distance between two edges

  # 25 cent min and max thresholds 
  d_25_min = 34 ** 2
  d_25_max = 36 ** 2

  # 5 cent min and max thresholds 
  d_5_min =  29 ** 2
  d_5_max =  32 ** 2

  # 1 cent min and max thresholds 
  d_1_min = 26 ** 2
  d_1_max = 28 ** 2

  # 10 cent min and max thresholds 
  d_10_min = 25 ** 2
  d_10_max = 26 ** 2

  # Distance of view, max distance in which the edges can exist. Mainly used for removing duplicate detetions.
  dov = 38
  dov_thr = dov ** 2


  # ----------- 2.2) Read and Resize the image--------------------------------------

  img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
  new_width = len(img[0])//downsize_factor
  new_height = len(img)//downsize_factor
  img = cv2.resize(img, (new_width, new_height))


  #-------- 2.3) Run canny and calculate valid edges ---------------------------------
  height, width = img.shape
  edges = cv2.Canny(img, 250, 300, 3,)

  # valid edges
  edge_pt = []
  for i in range(height):
    for j in range(width):
      if edges[i,j] > 127:
        edge_pt.append((i,j))




  #------------ 2.4)  compute Votes in the distance of view ---------------------------------

  # 2.4.1) Calculate the connected components from the edges, to determine the length of the edge
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)


  # This array stores the votes mapping to each center
  votes = np.zeros((height,width), np.int32)

  # this stores the votes for each coin denomination and each deetected center point
  coin_values = dict()

  # if valid edges found
  if len(edge_pt) != 0:

    # Iterate through all edges and compute the difference between each edge and all other edges.
    for i in range(len(edge_pt)):
      y1,x1 = edge_pt[i]

      for j in range(i+1, len(edge_pt)):
        y2,x2 = edge_pt[j]

        if abs(x2 - x1) > dov or abs(y2 - y1) > dov:
          continue

        # compute the eucledian distance between the two edge points
        d = (x1-x2)**2+(y1-y2)**2

        # if this distance is higher than our distance of view, discard this point and move on to the next one
        if d > dov_thr:
          continue

        # If the distance was in the valid distance of view, then calculate the center points of these two edge points
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        
        # if the length of this edge is too small, then discard it, possibly it's a texture
        c_label = labels[cy, cx]
        c_h_dist = stats[c_label, cv2.CC_STAT_WIDTH]
        c_v_dist = stats[c_label, cv2.CC_STAT_HEIGHT]

        c_dist = max(c_h_dist, c_v_dist)

        if c_dist <= 6:
          continue

        # if the length of the edges is big enough and the distance between these edges is within the valid distance of views, 
        # then first check if it is within any valid coin thresholds, if so give a vote to this center and give a vote to the corresponding denomination
        if d >= d_1_min and d <= d_1_max:
          votes[cy,cx] += 1

          if (cy, cx) not in coin_values:
            coin_values[(cy, cx)] = dict.fromkeys([1, 5, 25, 10], 0)
          coin_values[(cy, cx)][1] += 1

        elif d >= d_5_min and d <= d_5_max:
          votes[cy,cx] += 1

          if (cy, cx) not in coin_values:
            coin_values[(cy, cx)] = dict.fromkeys([1, 5, 25, 10], 0)
          coin_values[(cy, cx)][5] += 1

        elif d >= d_25_min and d <= d_25_max:
          
          votes[cy,cx] += 1

          if (cy, cx) not in coin_values:
            coin_values[(cy, cx)] = dict.fromkeys([1, 5, 25, 10], 0)
          coin_values[(cy, cx)][25] += 1


        elif d >= d_10_min and d <= d_10_max:
          votes[cy,cx] += 1

          if (cy, cx) not in coin_values:
            coin_values[(cy, cx)] = dict.fromkeys([1, 5, 25, 10], 0)
          coin_values[(cy, cx)][10] += 1

        else:
          continue

    # normalize the votes array to have the values between 0 and 1
    votes = votes/np.amax(votes)



  # ------------ 2.5) Remove redundant / less voted votes-------------------------------------

  # detect circles with this confidence
  isCoin_thr = 0.4

  # remove any redundancies in this radius
  d_thr = 30 ** 2

  # convert the grayscale image to color, for visually reviewing the detected circles   
  img2 = np.stack((img,img,img), axis=2)
  img2.shape

  # store the correctly marked coins in this list 
  marked = []

  height, width = img2.shape[:2]

  for i in range(height):
    for j in range(width):

        # if this is a highly voted center
        if votes[i,j] > isCoin_thr:

          flag = 0

          # check if a similar center was already marked, if so don't mark this center
          for (mi, mj) in marked:
              d = (i-mi)**2+(j-mj)**2
              if d <= d_thr:
                flag = 1
                break


          # if a center similar to this was not already marked, mark it
          if flag == 0:
              marked.append((i,j))
              cv2.circle(img2, (j,i), 10, (255,0,0), -1)




  #---------------------  3. Compare the results of the two approaches and dispy the output ---------------------------------

  # if edge detection detected more coins
  if len(marked) > len(valid_coins):
    print(len(marked))
    for (j, i) in marked:
        print('{} {} {}'.format(str(i*downsize_factor), str(j*downsize_factor), max(coin_values[(j, i)], key=coin_values[(j, i)].get)))

  # if Segmentation detected more coins
  else:
    print(len(valid_coins))
    for m_cx, m_cy, val in valid_coins:
      print(m_cx*downsize_factor, m_cy*downsize_factor, val)

# if Segmentation worked just display the output from segmentation 
else:
  print(len(valid_coins))
  for m_cx, m_cy, val in valid_coins:
    print(m_cx*downsize_factor, m_cy*downsize_factor, val)