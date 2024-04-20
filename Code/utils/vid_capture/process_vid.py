import cv2
import torch
import numpy as np
from PIL import Image


__all__ = ['ProcessVideo']


class ProcessVideo:
    """ Adds Bounding Boxes in a video. The video can be a live Capture or a saved video 
        depends on the cap argument in the __init__ function. 
    """

    def __init__(self, cap, transform, label_map, device, *, width=640, height=480,
                 downsample_factor=0.5, interpolation=cv2.INTER_LINEAR):

        self.cap = cap
        self.tranform = transform
        self.label_map = label_map
        self.device = device
        self.width = width
        self.height = height
        self.downsample_factor = downsample_factor
        self.interpolation = interpolation

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _process_frame(self, frame, detector, classifier):
        # Downsample frames (Faster detection)
        small_frame = cv2.resize(frame, None, fx=self.downsample_factor,
                                 fy=self.downsample_factor, interpolation=self.interpolation)
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces on rgb frame
        faces = detector(rgb_frame)

        for box in faces:
            # Scale bounding box coordinates back to original frame size
            x, y, x2, y2 = [int(coord / self.downsample_factor)
                            for coord in box]

            # Ensure coordinates are within frame bounds
            x, y, x2, y2 = max(0, x), max(0, y), min(
                frame.shape[1], x2), min(frame.shape[0], y2)

            # Extract face ROI, if not empty
            if x2 > x and y2 > y:
                face_roi = frame[y:y2, x:x2]
                face_pil = Image.fromarray(
                    cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

                # Apply transformations and add batch dimension
                face_transformed = self.tranform(
                    face_pil).unsqueeze(0).to(self.device)

                # Perform inference
                with torch.no_grad():
                    classifier = classifier.to(self.device)
                    output = classifier(face_transformed)
                    _, predicted = torch.max(output, 1)
                    label = self.label_map[predicted.item()]

                # Draw the bounding box and label
                cv2.rectangle(frame, (x, y), (x2, y2), (20, 10, 255), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (45, 25, 255), 2)

        return frame

    def offline(self, detector, classifier):
        # Count number of frames in the Video
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Initialize an empty video buffer
        video = np.empty(
            (frame_count, self.height, self.width, 3), np.dtype('uint8'))

        fc = 0
        ret = True
        # Loop the Video processing every frame.
        while fc < frame_count and ret:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(
                    frame, (self.width, self.height), interpolation=self.interpolation)
                frame = self._process_frame(frame, detector, classifier)
                video[fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fc += 1

        return video

    def live(self, detector, classifier):
        print('Press q to quit')
        ret = True
        while cv2.waitKey(1) & 0xFF != ord('q') and ret:
            ret, frame = self.cap.read()
            frame = self._process_frame(frame, detector, classifier)

            cv2.imshow('Face Mask Detection \n (Press q to quit)', frame)
