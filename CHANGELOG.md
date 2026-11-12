# Changelog

## 0.5.0

### New block-palette algorithm

`create_block_palette`, `extend_block_palette` and `create_theme_palette` now build
each block as a smooth curve in CAM02-UCS through a base colour, sampled evenly by
perceptual arc length, and choose the blocks' base colours knowing the block sizes.

- Blocks run from dark to light with strictly increasing lightness, no repeated
  colours, and even spacing (step variation under 5%, against 20-100% before),
  including under tight lightness bounds where the old method produced repeated
  colours and large hue jumps within a block.
- Large blocks are given room to span most of the lightness range; small blocks
  sit near middling lightness. Blocks are kept apart using a distance that
  discounts lightness, since lightness is what varies within a block.
- Block length grows with block size but more slowly, so small blocks get large
  steps: a pair is a clear dark and light version of one hue (step about 32),
  while a ten-colour block steps by about 10.
- Base colours favour rich, vivid colours near the saturated corners of RGB
  (reds, blues, greens first). Blocks keep their chroma through dark and middle
  lightness and ease into softer tints at their light end.
- A warm highlight shift: each block's light end turns toward yellow (reds
  lighten through orange toward gold, greens toward yellow-green), while blues
  rotate much less and mostly pale. The strength of the shift depends on the
  block's hue; blues get up to 20% more length (mostly lightness range) to stay
  as distinct within the block. Pairs keep a single hue.
- With `optimize_palette=True`, palettes are polished at the block level (sliding
  a block along its curve, slightly shrinking its step, or refitting a short block
  from a nudged base). Colours never leave their block or change order.

**Block palettes look different from 0.4.0.** This is intended. To reproduce
0.4.0 output exactly, pass `theme_method="legacy"` (see below).

### New parameters

All keyword-only and optional, on `create_block_palette` and `extend_block_palette`
(`theme_step` also on `create_theme_palette`):

- `theme_step` (default `None`): target CAM02-UCS distance between neighbouring
  colours in a block. `None` uses the size-dependent schedule above; a number uses
  that step for every block. Steps shrink automatically, with a warning below 4,
  when a block cannot fit in the bounds.
- `hue_sweep` (default `"auto"`): per-block hue rotation. A number caps the
  rotation in degrees; `0` disables it. With `colorblind_safe=True` the sweep is
  capped at 15 degrees.
- `twist` (default `"warm"`): the warm highlight shift described above. `1` or
  `-1` instead rotate every block the same way around the hue circle,
  symmetrically about its base; `"auto"` picks that shared direction.
- `theme_method` (default `"arc"`): `"legacy"` runs the 0.4.0 algorithm unchanged
  and emits a `FutureWarning`. It will be removed in a future release.

### Deprecations

- `theme_color_spacing` (block palettes) and `min_color_offset`
  (`create_theme_palette`) are deprecated. If passed, they map to
  `theme_step = 60 * value` (the old default 0.2 corresponds to 12) and emit a
  `DeprecationWarning`.

### Other changes

- `optimize_palette_n_iter` and `optimize_palette_search_radius` now control
  block-level polishing; the search radius is the maximum slide, in CAM02-UCS
  distance, along a block's curve.
- `extend_block_palette` is now listed in the API reference.
- Internal: curve primitives moved to `glasbey/_curves.py`, with a fast NumPy
  CAM02-UCS to sRGB conversion. `create_colormap` output is unchanged.
