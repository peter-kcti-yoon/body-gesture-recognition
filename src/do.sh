xhost local:root
docker run --privileged --rm -it \
	-v /dev/video0:/dev/video0 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=unix$DISPLAY \
	-v /dev/snd:/dev/snd \
	-e="QT_X11_NO_MITSHM=1" \
	-v $(pwd):/home intelli:mp-kinect-v1 /bin/bash
