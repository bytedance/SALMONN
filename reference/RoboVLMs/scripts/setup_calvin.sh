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

# sudo apt-get update -yqq

# Install dependency for calvin
# sudo apt-get -yqq install libegl1 libgl1

# Install EGL mesa
# sudo apt-get install -yqq libegl1-mesa libegl1-mesa-dev
# sudo apt-get install -yqq mesa-utils libosmesa6-dev llvm
# sudo apt-get -yqq install meson
# sudo apt-get -yqq build-dep mesa

conda install -c conda-forge gcc=12.1.0 gxx_linux-64 -y

git clone --recurse-submodules https://github.com/mees/calvin.git

CALVIN_ROOT=$(pwd)/calvin
cd ${CALVIN_ROOT}
sed -i '11d' calvin_models/requirements.txt
sed -i '12d' calvin_models/requirements.txt
sed -i '13d' calvin_models/requirements.txt
sed -i '14d' calvin_models/requirements.txt

sh install.sh

# CALVIN spesicifcally requires the following version of numpy
pip install numpy==1.21.0

# Download dataset
cd ${CALVIN_ROOT}/dataset
# sh download_data.sh debug
# sh download_data.sh D
# sh download_data.sh ABC
sh download_data.sh ABCD