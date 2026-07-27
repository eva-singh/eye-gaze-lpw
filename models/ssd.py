import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead


def get_model():

    model = torchvision.models.detection.ssd300_vgg16(
        weights="DEFAULT"
    )

    # Background + pupil
    num_classes = 2

    # Feature map channels (SSD300 VGG16)
    in_channels = [512, 1024, 512, 256, 256, 256]

    # Anchors on each feature map
    num_anchors = [4, 6, 6, 6, 4, 4]

    # Replace only the classifier
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    return model