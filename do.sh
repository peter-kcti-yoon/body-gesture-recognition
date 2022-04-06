xhost local:root
docker run --privileged --rm -it \
	-v /dev/video0:/dev/video0 \
	--gpus all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=unix$DISPLAY \
	-v /dev/snd:/dev/snd \
	-e="QT_X11_NO_MITSHM=1" \
	-p 6006:6006 \
	-v $(pwd):/home docker.inbee.i234.me/intelli:mediapipe-only-tf /bin/bash
	# -v $(pwd):/home docker.inbee.i234.me/intelli:10.1-cudnn7-ubuntu18.04 /bin/bash
	# -v $(pwd):/home docker.inbee.i234.me/intelli:cpp /bin/bash
