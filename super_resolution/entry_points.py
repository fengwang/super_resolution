"""
super_resolution.entry_points.py
~~~~~~~~~~~~~~~~~~~~~~

This module contains the entry-point functions for the super_resolution module,
that are referenced in setup.py.
"""


from sys import argv

from .super_resolution import cartoon_upsampling_4x


def main() -> None:
    """Main package entry point.

    Upsampling an input image by 4x.
    """
    try:
        input_image_path = argv[1]
        output_image_path = argv[2]

        cartoon_upsampling_4x( input_image_path, output_image_path )
    except IndexError:
        RuntimeError('Usage: INPUT_LOW_RESOLUTION_PATH OUTPUT_HIGH_RESOLUTION_PATH')
    return None

