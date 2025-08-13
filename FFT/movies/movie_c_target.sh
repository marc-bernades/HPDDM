#!/bin/sh

cd ./Cond_phase_speed_XY_pb/rho/c_1/

ls *.png | sort -V > ordered.txt



cat ordered.txt | while read img; do cat "$img"; done | ffmpeg -f image2pipe -r 12 -vcodec png -i - \
  -y -crf 18 -preset ultrafast -r 12 \
  -vcodec libx264 -s 1024x692 -pix_fmt yuv420p \
  -profile:v high444 \
  -metadata author="Marc Bernades" \
  -metadata copyright="Marc Bernades" \
  -metadata title="FFT_c_target" \
  -metadata comment="mailto:marc.bernades@cerfacs.fr" \
  -metadata year="2025-06-06" \
  ../../XY_c_plus_target_1_pb_rho.mp4
