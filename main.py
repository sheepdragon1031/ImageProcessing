from ImageProcessing.detect import detect
from ImageProcessing.profile import profile
from ImageProcessing.picture import TEST_IMAGE


def main():
    detect(profile(TEST_IMAGE['1']))


if __name__ == '__main__':
    main()
