# Live2D model source

The Cubism 3 model assets in this directory are the `AiPoetKurisu` lab-coat
model distributed by [Akatosh6DW/Amadeus](https://github.com/Akatosh6DW/Amadeus)
and attributed there to ai-poet's
[amadeus-system-new](https://github.com/ai-poet/amadeus-system-new) project.

The Akatosh6DW project states in its
[asset rights notice](https://github.com/Akatosh6DW/Amadeus/blob/main/ASSETS_LICENSE.md)
that the maintainer confirmed permission to use and distribute this model as
part of the Amadeus project. The model remains copyrighted by its original
author, is not covered by this repository's software license, and must not be
redistributed as a standalone asset.

The desktop model uses the neutral idle motion and a curated set of gestures
and expression motions. Motion files are configured as one-shots except for
the neutral idle pose, which loops. `random2.motion3.json` is intentionally
not referenced because the installed `live2d-py` runtime crashed during its
playback/cleanup test. Eye blinking and breathing are enabled through the
model's native Cubism parameters; the phone page continues to use its existing
model.
