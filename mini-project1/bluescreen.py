"""
File: bluescreen.py
--------------------
This program shows an example of "greenscreening" (actually
"bluescreening" in this case).  This is where we replace the
pixels of a certain color intensity in a particular channel
(here, we use blue) with the pixels from another image.
"""


from simpleimage import SimpleImage


INTENSITY_THRESHOLD = 1.6


def bluescreen(main_filename, back_filename):
    """
    Replaces blue-background pixels in main image with pixels from background image.
    """
    image = SimpleImage(main_filename)
    back = SimpleImage(back_filename)

    # Ensure the background image matches the size of the foreground image
    back.make_as_big_as(image)

    for pixel in image:
        if pixel.blue > INTENSITY_THRESHOLD * max(pixel.red, pixel.green):
            x, y = pixel.x, pixel.y
            back_pixel = back.get_pixel(x, y)
            pixel.red = back_pixel.red
            pixel.green = back_pixel.green
            pixel.blue = back_pixel.blue

    return image


def main():
    """
    Run your desired image manipulation functions here.
    You should store the return value (image) and then
    call .show() to visualize the output of your program.
    """
    original_stop = SimpleImage('./sample_foreground.png')
    original_stop.show()

    original_leaves = SimpleImage('./sample_background.png')
    original_leaves.show()

    stop_leaves_replaced = bluescreen('./sample_foreground.png', './sample_background.png')
    stop_leaves_replaced.show()
    stop_leaves_replaced.save('sample_with_bluescreen.png')

if __name__ == '__main__':
    main()
