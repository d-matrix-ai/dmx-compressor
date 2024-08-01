#!/usr/bin/env python3

import sys
from dmx import compressor
from dmx.compressor import *

sys.modules['mltools'] = compressor
