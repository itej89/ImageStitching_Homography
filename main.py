import os

import cv2 as cv
import numpy as np
import math

import matplotlib.pyplot as plt

#This library is used to fiter outliers from matched features
from skimage.measure import ransac
from skimage.transform import AffineTransform


#Funciton to find homography as part of ransac
def findHomography(scn_pts, obj_pts):
        
        A = []
        #Create linear equations
        for i in range(len(scn_pts)):
            x1,y1 = obj_pts[i]
            p1,q1 = scn_pts[i]
            A.append([-x1, -y1, -1, 0, 0, 0, x1*p1, y1*p1, p1])
            A.append([0, 0, 0, -x1, -y1, -1, x1*q1, y1*q1, q1])

        #Solve linear equations using SVD
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1, :] / Vt[-1, -1]
        H = H.reshape(3, 3)

        
        return H

#method to stitch images by calculating homography
def stitch_images(im1, im2, ransac_threshold, draw_matches = False):

    #Convert images to gray scale---------------------
    im1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    #-------------------------------------------------


    #Run SIFT to find features-------------------------------
    sift = cv.SIFT_create()
    im1_keypoints, im1_descriptors = sift.detectAndCompute(im1_gray,None)
    im2_keypoints, im2_descriptors = sift.detectAndCompute(im2_gray,None)
    #--------------------------------------------------------------------

    

    #show key points-------------
    # output_image1 = cv.drawKeypoints(im1_gray, im1_keypoints, 0, (0, 0, 255),
    #                              flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # output_image2 = cv.drawKeypoints(im2_gray, im1_keypoints, 0, (0, 0, 255),
    #                              flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # im_key_points = np.hstack((output_image1, output_image2))
    # plt.imshow(im_key_points)
    # plt.show()
    #-----------------------------------------------------



    #Run knn matcher to find the matches--q--------------------------------------------------
    brute_froce_matcher = cv.BFMatcher()
    matches = brute_froce_matcher.knnMatch(im1_descriptors,im2_descriptors,k=1)
    #---------------------------------------------------------------------------

    #extract the matched point pairs-------------------------------------------------------
    scn_pts = np.array([np.round(im1_keypoints[match[0].queryIdx].pt).astype(np.float32).tolist() for match in matches])
    obj_pts = np.array([np.round(im2_keypoints[match[0].trainIdx].pt).astype(np.float32).tolist() for match in matches])
    #--------------------------------------------------------------------------------------
    

    #Run RANSAC to filter the outliers in matched points----------------------------------------
 
    inliers = []
    n_inliers = 0
    while n_inliers < 20:
        _, inliers = ransac((scn_pts, obj_pts),AffineTransform, min_samples=4,residual_threshold=ransac_threshold, max_trials=10000)
        n_inliers = np.sum(inliers)
        if n_inliers < 20:
            print(f"Found low number of inliers : {n_inliers}, recomputing RANSAC. Please wait...")

    inlier_keypoints_left = [cv.KeyPoint(point[0], point[1], 1) for point in scn_pts[inliers]]
    inlier_keypoints_right = [cv.KeyPoint(point[0], point[1], 1) for point in obj_pts[inliers]]
    ransac_matches = [cv.DMatch(idx, idx, 1) for idx in range(n_inliers)]

   
    # show filtered matches-------------
    if draw_matches:
        img_feature_matched = cv.drawMatches(im1_gray, inlier_keypoints_left, im2_gray, inlier_keypoints_right, ransac_matches, None)
        cv.imshow("img_feature_matched", img_feature_matched)
        cv.waitKey(0)
    #---------------------------------------------------------------------------

    
    #extract the matched point pairs from filtererd points-----------------------
    scn_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in ransac_matches ])
    obj_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in ransac_matches ])
    #---------------------------------------------------------------------------


    #Find best homography matrix through Ransac algorithm-------------------------------------
    max_inlier_count = 0
    best_inlier_indices = []
    

    #FIND HOMOGRAPHY USING RANSAC----------------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    itr=  0
    while itr<10:
        itr += 1

        #sample points
        #Chosen the value 100 through trial and error and realiability
        random_idx = np.random.choice(len(scn_pts), 100)
        sample_scn_pts = scn_pts[random_idx]
        sample_obj_pts = obj_pts[random_idx]

        H = findHomography(sample_scn_pts, sample_obj_pts)
        inlier_count = 0
        inlier_indices = []

        #estimate inliers using the computed homography---------
        for i in range(len(scn_pts)):
            X = scn_pts[i].copy()

            #make a homogrnous coordinate
            P = np.append(obj_pts[i].copy(), 1)
            P = P.reshape(3, 1)

            #predict the scn point
            Xpred = H @ P
            Xpred_hmg = np.squeeze((Xpred/Xpred[2])[:2])

            #Compute differance between prediction and actual feature location
            Xdiff = (Xpred_hmg - X)
            Xerror =  np.linalg.norm(Xdiff)

            #count inliers
            if Xerror < 10:
                inlier_count +=1
                inlier_indices.append(i)

        #record highest inliers found so far
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_inlier_indices = inlier_indices.copy()



    #compute final homography using the best inliers discovered
    H = findHomography(scn_pts[best_inlier_indices], obj_pts[best_inlier_indices])
    #---------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------



    #warp perpective of the object image(image2) which needs to be projected on scene image(image1)
    result = cv.warpPerspective(im2, H, ((im1.shape[1] + im2.shape[1]), im2.shape[0])) 

    #paste  the image 1 to complete the stitching
    result[0:im1.shape[0], 0:im1.shape[1]] = im1

    return result


if __name__ =="__main__":

    #read images
    im1_path = os.path.join(os.path.dirname(__file__), "data", "image_1.jpg")
    im2_path = os.path.join(os.path.dirname(__file__), "data", "image_2.jpg")
    im3_path = os.path.join(os.path.dirname(__file__), "data", "image_3.jpg")
    im4_path = os.path.join(os.path.dirname(__file__), "data", "image_4.jpg")

    im1 = cv.imread(im1_path)
    im2 = cv.imread(im2_path)
    im3 = cv.imread(im3_path)
    im4 = cv.imread(im4_path)

    #scale images to save computation time
    scale_factor = 30
    width  = int(im1.shape[1] * scale_factor / 100)
    height = int(im1.shape[0] * scale_factor / 100)
    dim = (width, height)

    im1 = cv.resize(im1, dim, interpolation = cv.INTER_AREA)
    im2 = cv.resize(im2, dim, interpolation = cv.INTER_AREA)
    im3 = cv.resize(im3, dim, interpolation = cv.INTER_AREA)
    im4 = cv.resize(im4, dim, interpolation = cv.INTER_AREA)

    #stitch images in sequence-----------------------------------
    print("Stitching image1 and image2. Please wait...")
    output = stitch_images(im1, im2, ransac_threshold=8, draw_matches=False)
    print("Completed stitching image2.")
    output = output[:,:1108,:]
    print("Stitching image3. Please wait...")
    output = stitch_images(output, im3, ransac_threshold=8, draw_matches=False)
    print("Completed stitching image3.")
    output = output[:,:1310,:]
    print("Stitching image4. Please wait...")
    output = stitch_images(output, im4,ransac_threshold=8, draw_matches=False)
    print("Completed stitching image4.")
    #-----------------------------------------------------------

    cv.imshow("Final stitched panaromic image", output)
    cv.waitKey(0)

