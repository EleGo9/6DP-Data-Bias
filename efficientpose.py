from EfficientPose.model import build_EfficientPose

NUM_CLASSES = 3

def build_model_and_load_weights(phi, num_classes, score_threshold, path_to_weights):
    """
    Builds an EfficientPose model and init it with a given weight file
    Args:
        phi: EfficientPose scaling hyperparameter
        num_classes: The number of classes
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        path_to_weights: Path to the weight file

    Returns:
        efficientpose_prediction: The EfficientPose model
        image_size: Integer image size used as the EfficientPose input resolution for the given phi

    """

    print("\nBuilding model...\n")
    efficientpose_train, _, _ = build_EfficientPose(phi,
                                                         num_classes=num_classes,
                                                         num_anchors=9,
                                                         freeze_bn=True,
                                                         score_threshold=score_threshold,
                                                         num_rotation_parameters=3,
                                                         print_architecture=False)

    print("\nDone!\n\nLoading weights...")
    efficientpose_train.load_weights(path_to_weights, by_name=True)
    print("Done!")

    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]

    return efficientpose_train, image_size
