"""
影像處理主體
"""
from ImageProcessing.detect import detect
from ImageProcessing.profile import profile
from ImageProcessing.fillHole import fillHole
from ImageProcessing.files import MATERIAL, CASCADE_FILE, TEST_IMAGE, IMAGES

__all__ = [
    'detect', 'profile', 'fillHole', 'MATERIAL', 'CASCADE_FILE', 'TEST_IMAGE',
    'IMAGES'
]
