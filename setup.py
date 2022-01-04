import os
import re

import setuptools


def read(filename):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
    with open(path , 'r') as f:
      return f.read()


def find_version(text):
    match = re.search(r"^__version__\s*=\s*['\"](.*)['\"]\s*$", text,
                      re.MULTILINE)
    return match.group(1)


AUTHOR = "Conservation Technology Lab at the San Diego Zoo Wildlife Alliance"
DESC = "A package of ML components for CTL field camera systems. "

setuptools.setup(
    name="camml",
    description=DESC,
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    license="MIT",
    version=find_version(read('camml/__init__.py')),
    author=AUTHOR,
    packages=['camml'],
    include_package_data=True,
    install_requires=[
        'opencv-python',
        'pillow',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
    ],
)
