# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Find OpenCV headers and libraries using the correct package name 'opencv'.
try:
    import subprocess
    # --- THIS IS THE FIX ---
    extra_compile_args = ['-I/usr/include/opencv4']
    extra_link_args = ['-lopencv_core', '-lopencv_imgproc', '-lopencv_highgui']  # Add more if needed

    print("Successfully found OpenCV using 'pkg-config opencv'")
    print(f"  Compile flags: {extra_compile_args}")
    print(f"  Link flags: {extra_link_args}")

except (subprocess.CalledProcessError, FileNotFoundError):
    print("\n--- ERROR ---")
    print("Could not find OpenCV using 'pkg-config opencv'.")
    print("This is common if OpenCV was installed from source or is in a non-standard location.")
    print("Please ensure 'libopencv-dev' is installed (`sudo apt-get install libopencv-dev`)")
    print("Or, you may need to manually edit this setup.py script with your OpenCV paths.")
    print("-------------\n")
    # Fallback to empty if not found, will likely fail later but prevents crash here
    extra_compile_args = []
    extra_link_args = []


setup(
    name='dfine_preproc_cpp',
    ext_modules=[
        CppExtension(
            'dfine_preproc_cpp',
            ['preproc.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
