#!/usr/bin/env bash
set -euo pipefail

# Render a short, silent submission video from the actual saved policy rollout.
input=${1:-outputs/muse_dataset50_episode0000_v1_rollout.mp4}
output=${2:-outputs/object_dynamics_vla_hackathon_demo.mp4}
font=${DEMO_FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf}
bold=${DEMO_BOLD_FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i 'color=c=0x0b1120:s=1280x720:r=20:d=18.35' -i "$input" \
  -filter_complex "
    [1:v]scale=630:630:flags=lanczos,setpts=PTS-STARTPTS+6/TB[rollout];
    [0:v][rollout]overlay=600:45:eof_action=pass:enable='between(t,6,14.35)'[base];
    [base]drawtext=fontfile='$bold':text='OBJECT DYNAMICS VLA':fontsize=38:fontcolor=0x63e6d0:x=65:y=60,
      drawtext=fontfile='$bold':text='Put the milk in the basket':fontsize=56:fontcolor=white:x=65:y=265:enable='between(t,0,3)',
      drawtext=fontfile='$font':text='One instruction. A learned robot action policy.':fontsize=27:fontcolor=0xc5d1e2:x=68:y=350:enable='between(t,0,3)',
      drawtext=fontfile='$bold':text='CAMERA + INSTRUCTION':fontsize=38:fontcolor=white:x=68:y=215:enable='between(t,3,6)',
      drawtext=fontfile='$font':text='Frozen NF4 Muse features + SAM object tracking':fontsize=30:fontcolor=0xc5d1e2:x=68:y=290:enable='between(t,3,6)',
      drawtext=fontfile='$font':text='Query-normalized prefix  >  SmolVLA action expert':fontsize=28:fontcolor=0xc5d1e2:x=68:y=345:enable='between(t,3,6)',
      drawtext=fontfile='$bold':text='ACTUAL POLICY':fontsize=32:fontcolor=white:x=65:y=235:enable='between(t,6,14.35)',
      drawtext=fontfile='$bold':text='SIMULATOR ROLLOUT':fontsize=32:fontcolor=white:x=65:y=280:enable='between(t,6,14.35)',
      drawtext=fontfile='$font':text='Muse + SAM conditioned':fontsize=24:fontcolor=0xc5d1e2:x=65:y=350:enable='between(t,6,14.35)',
      drawtext=fontfile='$font':text='Checkpoint 50  |  seed 7':fontsize=23:fontcolor=0xc5d1e2:x=65:y=390:enable='between(t,6,14.35)',
      drawtext=fontfile='$font':text='Demonstrated training initialization':fontsize=20:fontcolor=0xc5d1e2:x=65:y=445:enable='between(t,6,14.35)',
      drawtext=fontfile='$bold':text='SIMULATOR SUCCESS':fontsize=56:fontcolor=0x63e6d0:x=65:y=235:enable='gte(t,14.35)',
      drawtext=fontfile='$font':text='Milk placed in basket after 166 actions':fontsize=32:fontcolor=white:x=68:y=315:enable='gte(t,14.35)',
      drawtext=fontfile='$font':text='1 of 3 sampling seeds succeeded on this training initialization':fontsize=24:fontcolor=0xc5d1e2:x=68:y=390:enable='gte(t,14.35)',
      format=yuv420p[out]" \
  -map '[out]' -an -c:v libx264 -crf 19 -preset medium -movflags +faststart "$output"
