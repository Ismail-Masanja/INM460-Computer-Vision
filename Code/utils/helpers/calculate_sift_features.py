import cv2


__all__ = ['calculate_sift_features']


def calculate_sift_features(img):
    # Create SIFT object
    sift = cv2.SIFT_create()
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(img, None) 

    return keypoints, descriptors
