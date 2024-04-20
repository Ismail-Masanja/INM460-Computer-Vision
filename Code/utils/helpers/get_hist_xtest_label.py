import numpy as np
from utils.helpers import image_preprocess, calculate_sift_features


__all__ = ['get_hist_xtest_label']


def get_hist_xtest_label(dataloader, kmeans, k):
    hist_list = []
    X_test = []
    test_labels = []

    for batch_images, labels in dataloader:
        for (image, label) in zip(batch_images, labels):
            # Preprocess the image
            img = image.permute(1, 2, 0).detach().cpu().numpy()
            X_test.append(img)
            img = image_preprocess(img)
            test_labels.append(label.item())
            keypoints, descriptors = calculate_sift_features(img)

            if descriptors is not None:
                hist = np.zeros(k)

                idx = kmeans.predict(descriptors)

                for j in idx:
                    hist[j] = hist[j] + (1 / len(descriptors))

                # hist = scale.transform(hist.reshape(1, -1))
                hist_list.append(hist)

            else:
                hist_list.append(None)

    return hist_list, X_test, test_labels
