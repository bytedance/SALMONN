# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/baaivision/UniVLA. The original license is located at 'third-party-license/UniVLA.txt'.

cd ..

LIBERO_ROOT="${pwd}/LIBERO"
if [ ! -d "$LIBERO_ROOT" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd "${LIBERO_ROOT}"
    pip install -e .
    pip install -r ../RoboVLMs/eval/libero/libero_requirements.txt
fi
