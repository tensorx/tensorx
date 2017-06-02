# Copyright 2017 Davide Nunes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================================================================
import tensorflow as tf


class Layer:
    def __init__(self, n_units, shape, dtype=tf.float32, name="layer"):
        self.n_units = n_units
        self.name = name
        self.tensor = None
        self.dtype = dtype

        if shape is None:
            self.shape = [None, n_units]
        elif shape[1] != n_units:
            raise Exception("Shape must match [,n_units], was " + shape)

    def tensor(self):
        return self.tensor


class Input(Layer):
    def __init__(self, n_units, dtype=tf.float32, name="input"):
        super().__init__(n_units, [None, n_units], dtype, name)
        self.tensor = tf.placeholder(self.dtype, self.shape, self.name)
