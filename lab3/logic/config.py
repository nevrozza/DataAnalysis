from enum import StrEnum


class Paths(StrEnum):
    OUTPUT_DIR = "output/"
    TEMP_DIR = OUTPUT_DIR+"temp/"
    SOURCE_DIR = "source/"

    ANNOTATIONS_DIR = OUTPUT_DIR+"annotations"
    IMAGES_DIR = TEMP_DIR+"images"

    SOURCE_ANNOTATIONS = SOURCE_DIR+"annotations.xml"
    SOURCE_INPUT = SOURCE_DIR+"input.mp4"
    SOURCE_OUTPUT = SOURCE_DIR+"output.mp4"

    YOLO_DIST = TEMP_DIR + "yolov8n.pt"
    BEST_MODEL_DIST =  OUTPUT_DIR + "best_model.pth"
    RESULT_VIDEO_DIST =  OUTPUT_DIR + "result.mp4"