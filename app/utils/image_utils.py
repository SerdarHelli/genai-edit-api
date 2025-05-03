
import io
import numpy as np

from fastapi import UploadFile, HTTPException
from PIL import Image
from loguru import logger
import cv2


def read_image_bytes(upload_file: UploadFile) -> Image.Image:
    """
    Read and convert an uploaded image file to RGB PIL Image.

    Args:
        upload_file (UploadFile): The uploaded image file.

    Returns:
        Image.Image: The processed RGB image.

    Raises:
        HTTPException: If the image cannot be read or is invalid.
    """
    try:
        image_data = upload_file.file.read()
        logger.info(f"Image '{upload_file.filename}' read successfully.")
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to read image '{upload_file.filename}': {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")


def to_canny(img: Image.Image) -> Image.Image:
    """
    Convert a PIL image to a Canny edge map.

    Args:
        img (Image.Image): Input PIL image.

    Returns:
        Image.Image: Output edge map as a PIL image.
    """
    np_img = np.array(img)
    edges = cv2.Canny(np_img, 100, 200)
    return Image.fromarray(edges)


def get_diff_mask(base: Image.Image, annotated: Image.Image) -> Image.Image:
    """
    Compute a binary mask of visual differences between two images.

    Args:
        base (Image.Image): Baseline image.
        annotated (Image.Image): Annotated/edited image.

    Returns:
        Image.Image: Binary mask highlighting visual differences.
    """
    base_np = np.array(base.convert("L")).astype(np.int16)
    ann_np = np.array(annotated.convert("L")).astype(np.int16)
    diff = np.abs(base_np - ann_np)
    _, thresh = cv2.threshold(diff.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)
