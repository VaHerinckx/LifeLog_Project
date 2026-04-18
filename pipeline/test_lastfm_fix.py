#!/usr/bin/env python3
"""
Test script to run Last.fm pipeline with option 1 automatically
"""
import sys
sys.path.insert(0, 'src')

from music.lastfm_processing import full_lfm_pipeline

# Run pipeline with auto_full=True to skip user input
success = full_lfm_pipeline(auto_full=True)

if success:
    print("\n✅ Test completed successfully!")
    sys.exit(0)
else:
    print("\n❌ Test failed!")
    sys.exit(1)
