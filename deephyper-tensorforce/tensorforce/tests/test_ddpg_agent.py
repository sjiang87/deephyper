# Copyright 2017 reinforce.io. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest

from tensorforce.tests.base_agent_test import BaseAgentTest
from tensorforce.agents import DDPGAgent


class TestDDPGAgent(BaseAgentTest, unittest.TestCase):

    agent = DDPGAgent
    config = dict(
        update_mode=dict(
            unit='timesteps',
            batch_size=8,
            frequency=8
        ),
        memory=dict(
            type='replay',
            include_next_states=True,
            capacity=100
        ),
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        critic_network=dict(
            size_t0=64,
            size_t1=64
        ),
        target_sync_frequency=10
    )
    exclude_multi = True
