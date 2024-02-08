# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adapted/copied from:

https://github.com/google-coral/tflite/blob/master/python/examples/detection/detect_image.py

But we should consider/try getting this from the following as there
are better docstrings (but it's implemented differently hence the need
to check/test first):

https://github.com/google-coral/pycoral/blob/master/pycoral/adapters/detect.py

"""
import platform

import tflite_runtime.interpreter as tflite

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def make_interpreter(model_file):
    """Prepare interpreter from model file

    """
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                 {'device': device[0]} if device else {})
        ])
