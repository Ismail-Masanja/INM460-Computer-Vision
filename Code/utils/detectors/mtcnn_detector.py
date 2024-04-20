from facenet_pytorch import MTCNN


__all__ = ['mtcnn_detector']


def mtcnn_detector(device, frame):
    """ Uses mtcnn from facenet to detect and calculate bounding boxes in a frame"""
    mtcnn = MTCNN(keep_all=True, device=device)
    boxes, _ = mtcnn.detect(frame)
    # Convert boxes from float to int, and reshape
    if boxes is not None:
        boxes = boxes.astype(int)
        # Reshape boxes to match format: (x, y, x2, y2)
        return [(box[0], box[1], box[2], box[3]) for box in boxes]
    else:
        return []
