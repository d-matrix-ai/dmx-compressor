#!/usr/bin/env python3

# test to make sure the old mltools api is packaged and an alias for dmx.compressor

def test_mltools_alias_dmx_compressor():
    import mltools
    import dmx.compressor

    assert mltools.dmx.__file__ == dmx.compressor.dmx.__file__
