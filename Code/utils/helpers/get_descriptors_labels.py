from utils.helpers import calculate_sift_features, image_preprocess


__all__ = ['get_descriptor_labels']


def get_descriptor_labels(dataloader):
    """ Preprocessing and Descriptor Extraction """
    all_descriptors = []
    all_labels = []
    for batch_images, labels in dataloader:
        for (image, label) in zip(batch_images, labels):
            # Preprocess the image
            img = image.permute(1, 2, 0).detach().cpu().numpy()
            img = image_preprocess(img)

            # Calculate SIFT features
            _, descriptors = calculate_sift_features(img)
            if descriptors is not None:
                all_descriptors.append(descriptors)
                all_labels.append(label)

    return all_descriptors, all_labels
