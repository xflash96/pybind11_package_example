DATA_VOLUME="-v $(pwd)/..:/data"
HOME_VOLUME="-v $HOME:$HOME"
docker run --rm --privileged -it --net=host --ipc=host ${DATA_VOLUME} ${HOME_VOLUME} -v /etc/localtime:/etc/localtime:ro example_docker_env bash -l
