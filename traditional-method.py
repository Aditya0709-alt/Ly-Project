import numpy as np
import cv2



def nms(image, corners, tolerance, num_best_corners):
    corners_boolean = corners > tolerance * corners.max()           
    coordinates_tuple = np.nonzero(corners_boolean)                 
    corners_locations = np.asarray(coordinates_tuple)               
    corners_locations = np.transpose(corners_locations)             
    num_corners = len(corners_locations)
    print('Number of corners:  ' + str(num_corners))                   

    r = np.ones((num_corners, 1)) * np.inf              
    ED = np.inf                                         

    for i in range(num_corners):
        for j in range(num_corners):
            x_i = corners_locations[i][1]                     
            y_i = corners_locations[i][0]                     
            x_j = corners_locations[j][1]                     
            y_j = corners_locations[j][0]

            C_i = corners[y_i, x_i]                           
            C_j = corners[y_j, x_j]

            if C_j > C_i:                                     
                ED = ((x_j - x_i) ** 2) + ((y_j - y_i) ** 2) 

            if ED < r[i]:                                     
                r[i] = ED                                                 

    
    corners_locations_np = np.array(corners_locations)                      
    corners_y, corners_x = np.split(corners_locations_np, 2, axis=1)        
    corners_ranked = np.concatenate((corners_y, corners_x, r), axis=1)      

     
    corners_sorted = corners_ranked[corners_ranked[:, 2].argsort()[::-1]]   

    if num_best_corners > num_corners:
        num_best_corners = num_corners
    corners_best = corners_sorted[0:num_best_corners, :]                    

    
    for i in range(num_best_corners):
        x = int(corners_best[i, 1])
        y = int(corners_best[i, 0])
        green = [0, 255, 0]                                                
        image[y, x] = green                                                


    return corners_best


def get_features(points_best, grey_image):
    patch_size = 40                                 
    features = np.array(np.zeros((64, 1)))          

    grey_image = np.pad(grey_image, int(patch_size), 'constant', constant_values=0)    

    num_features, cols = points_best.shape

    for i in range(num_features):

        patch_y = points_best[i][0] + patch_size
        patch_x = points_best[i][1] + patch_size

        patch = grey_image[int(patch_y - (patch_size / 2)):int(patch_y + (patch_size / 2)),
                int(patch_x - (patch_size / 2)):int(patch_x + (patch_size / 2))]

        patch_blurred = cv2.GaussianBlur(patch, (5, 5), 0)          
        patch_sub_sampled = cv2.resize(patch_blurred, (8, 8))       

        patch_sub_sampled = patch_sub_sampled.reshape(64, 1)        

        patch_sub_sampled = (patch_sub_sampled - np.mean(patch_sub_sampled)) / np.std(patch_sub_sampled)

        features = np.dstack((features, patch_sub_sampled))         

    cv2.imwrite('blurred.png',patch_blurred)
     
    print(features[:, :, 1:])
    return features[:, :, 1:]  
    


def match_features(features1, features2, corners1, corners2):
    
    a, b, num_features1 = features1.shape
    c, d, num_features2 = features2.shape
    min_features = int(min(num_features1, num_features2))
    max_features = int(max(num_features1, num_features2))

    tolerance_diff_ratio = 0.7                             
    match_pairs = []                                       

    
    for i in range(min_features):

        matches = {}    

        for j in range(max_features):

            feature1 = features1[:, :, i]
            feature2 = features2[:, :, j]
            corner1 = corners1[i, :]
            corner2 = corners2[j, :]

            diff_sum_squares = np.linalg.norm((feature1 - feature2)) ** 2       
            matches[diff_sum_squares] = [corner1, corner2]

        sorted_matches = sorted(matches)    
        if sorted_matches[0] / sorted_matches[1] < tolerance_diff_ratio:
            pairs = matches[sorted_matches[0]]
            match_pairs.append(pairs)

    return match_pairs


def visualize_matches(image1, image2, matched_pairs):
    if len(image1.shape) == 3:
        height1, width1, depth1 = image1.shape
        height2, width2, depth2 = image2.shape
        shape = (max(height1, height2), width1 + width2, depth1)

    elif len(image1.shape) == 2:
        height1, width1, depth1 = image1.shape
        height2, width2, depth2 = image2.shape
        shape = (max(height1, height2), width1 + width2)

    image_combined = np.zeros(shape, type(image1.flat[0]))          
    image_combined[0:height1, 0:width1] = image1                    
    image_combined[0:height1, width1:width1 + width2] = image2      
    image_12 = image_combined.copy()

    circle_size = 4
    red = [0, 0, 255]
    cyan = [255, 255, 0]
    yellow = [0, 255, 255]

    
    for i in range(len(matched_pairs)):

        corner1_x = matched_pairs[i][0][1]
        corner1_y = matched_pairs[i][0][0]
        corner2_x = matched_pairs[i][1][1]
        corner2_y = matched_pairs[i][1][0]

        cv2.line(image_12, (int(corner1_x), int(corner1_y)), (int(corner2_x + image1.shape[1]), int(corner2_y)), red, 1)
        cv2.circle(image_12, (int(corner1_x), int(corner1_y)), circle_size, cyan, 1)
        cv2.circle(image_12, (int(corner2_x) + image1.shape[1], int(corner2_y)), circle_size, yellow, 1)

    
    cv2.imwrite("match.png", image_12)
    


def RANSAC(matches):
    

    iterations = 10000                              
    tau = 30                                        
    num_matches = len(matches)                      
    percent_good_matches = 0.6                     
    latest_homography = np.zeros((3, 3))            
    maximum = 0

    for index in range(iterations):                 

        pairs_indices = []                          

        
        points = [np.random.randint(0, num_matches) for num in range(4)]      

        pt_1 = np.flip(matches[points[0]][0][0:2])          
        pt_2 = np.flip(matches[points[1]][0][0:2])          
        pt_3 = np.flip(matches[points[2]][0][0:2])
        pt_4 = np.flip(matches[points[3]][0][0:2])
        pt_prime_1 = np.flip(matches[points[0]][1][0:2])
        pt_prime_2 = np.flip(matches[points[1]][1][0:2])
        pt_prime_3 = np.flip(matches[points[2]][1][0:2])
        pt_prime_4 = np.flip(matches[points[3]][1][0:2])

        pts = np.array([pt_1, pt_2, pt_3, pt_4], np.float32)                                    
        pts_prime = np.array([pt_prime_1, pt_prime_2, pt_prime_3, pt_prime_4], np.float32)

        H = cv2.getPerspectiveTransform(pts, pts_prime)

        num_good_matches = 0            

        
        for i in range(num_matches):

            y_pt = matches[i][0][0]
            x_pt = matches[i][0][1]
            y_pt_prime = matches[i][1][0]
            x_pt_prime = matches[i][1][1]

            pt_prime_i = np.array([x_pt_prime, y_pt_prime])     

            pt_i = np.array([x_pt, y_pt, 1])                    
            H_pt_i = np.matmul(H, pt_i)                         

            if H_pt_i[2] == 0:                                  
                H_pt_i[2] = 0.0000001
            p_x = H_pt_i[0] / H_pt_i[2]                         
            p_y = H_pt_i[1] / H_pt_i[2]

            H_pt_i = np.array([p_x, p_y], np.float32)
            

            SSD = np.linalg.norm((pt_prime_i - H_pt_i)) ** 2          

            if SSD < tau:                                             
                num_good_matches += 1
                pairs_indices.append(i)

        matches_p = []
        matches_prime = []

        if maximum < num_good_matches:
            maximum = num_good_matches

            [matches_p.append([np.flip(matches[ind][0][0:2])]) for ind in pairs_indices]
            [matches_prime.append([np.flip(matches[ind][1][0:2])]) for ind in pairs_indices]
            

            latest_homography, a = cv2.findHomography(np.float32(matches_p), np.float32(matches_prime))

            if num_good_matches > percent_good_matches * num_matches:       
                break

    matched_pairs = [matches[i] for i in pairs_indices]         

    return latest_homography, matched_pairs


def combine_images(image1, homography1to2, image2):
    
    
    image2_shape = image2.shape

    h, w, k = np.shape(image1)
    random_H = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    H = np.dot(homography1to2, random_H)

    row_y = H[1] / H[2]
    row_x = H[0] / H[2]

    new_mat = np.array([[1, 0, -1 * min(row_x)], [0, 1, -1 * min(row_y)], [0, 0, 1]])
    homography1to2 = np.dot(new_mat, homography1to2)

    h = int(round(max(row_y) - min(row_y))) + image2_shape[0]
    w = int(round(max(row_x) - min(row_x))) + image2_shape[1]
    size = (h, w)

    image1_warped = cv2.warpPerspective(src=image1, M=homography1to2, dsize=size)
    return image1_warped, int(min(row_x)), int(min(row_y))


def main():
    
    im1_path = '{PATH_TO_IMAGE-1}'
    im2_path = '{PATH_TO_IMAGE-2}'
    


    image1 = cv2.imread(im1_path)                      
    image2 = cv2.imread(im2_path)
    print()
    image1_grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)          
   
    image2_grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    

    
    block_size = 2      
    k_size = 3          
    k = 0.05            
    image1_dst = cv2.cornerHarris(image1_grey, block_size, k_size, k)
    image2_dst = cv2.cornerHarris(image2_grey, block_size, k_size, k)
    tolerance = 0.04 
    image1_corners = image1.copy()
    image2_corners = image2.copy()
    
    purple = [255, 0, 255]  
    image1_corners[image1_dst > tolerance * image1_dst.max()] = purple
    image2_corners[image2_dst > tolerance * image2_dst.max()] = purple
    
    num_best = 100
    image1_corners_best = nms(image1_corners, image1_dst, tolerance, num_best)
    image2_corners_best = nms(image2_corners, image2_dst, tolerance, num_best)

    image1_nms = image1.copy()
    yellow = [0, 255, 255]
    dot_size = 9

    for corner in image1_corners_best:                                  
        x = int(corner[1])
        y = int(corner[0])
        cv2.circle(image1_nms, (x, y), dot_size, yellow, -1)
        cv2.imwrite('nms.png',image1_nms)
    
    


    
    image1_features = get_features(image1_corners_best, image1_grey)
    image2_features = get_features(image2_corners_best, image2_grey)
    print(image1_features.shape)

    

    matched_1_2 = match_features(image1_features, image2_features, image1_corners_best, image2_corners_best)
    
    H_1_2, matched_1_2 = RANSAC(matched_1_2)

    visualize_matches(image1, image2, matched_1_2)
    
    im = image2.copy()
    im2 = image1.copy()


    inv_H_1_2 = np.linalg.inv(H_1_2)

    panorama, offsetX, offsetY = combine_images(im2, H_1_2, im)
    
    print("panorama", panorama.shape)
    print("y: ", im.shape[0] + abs(offsetY))
    print("x:", im.shape[1] + abs(offsetX))

    for y in range(abs(offsetY), im.shape[0] + abs(offsetY)):
        for x in range(abs(offsetX), im.shape[1] + abs(offsetX)):
            img2_y = y - abs(offsetY)
            img2_x = x - abs(offsetX)
            panorama[y, x, :] = im[img2_y, img2_x, :]
    cv2.imwrite('panaroma.png',panorama)

    im2 = panorama


if __name__ == "__main__":
    main()
