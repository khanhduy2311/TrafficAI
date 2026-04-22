# setup_wbf.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

print("Building 'wbf_cpp' extension...")
# This build is much simpler as it does not need OpenCV.
setup(
    name='wbf_cpp',
    ext_modules=[
        CppExtension('wbf_cpp', ['wbf.cpp'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
