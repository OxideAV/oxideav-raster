#!/bin/sh
# Regenerate the black-box reference rasters for tests/fixtures/svg/*.svg.
#
# References are produced by rsvg-convert (librsvg 2.62 / cairo 1.18 at
# the time of writing) and converted to raw straight-alpha RGBA bytes with
# ImageMagick so the test harness needs no image decoder. Every fixture is
# 64×64 pixels; the file name carries the size.
#
#   sh tests/fixtures/gen.sh
set -eu
cd "$(dirname "$0")"
for svg in svg/*.svg; do
  name=$(basename "$svg" .svg)
  rsvg-convert -w 64 -h 64 "$svg" -o "ref/$name.png"
  magick "ref/$name.png" -depth 8 "rgba:ref/${name}_64x64.rgba"
  rm "ref/$name.png"
done
echo "regenerated $(ls ref/*.rgba | wc -l | tr -d ' ') references"
