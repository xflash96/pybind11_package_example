FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
ARG HOME_DIR

RUN apt-get -q update; apt-get -q -y install sudo vim git
RUN apt-get install -y linux-tools-common linux-tools-generic linux-cloud-tools-generic
RUN conda install -y line_profiler

RUN addgroup --gid ${GROUP_ID} ${USER_NAME} || groupmod -n ${USER_NAME} $(getent group ${GROUP_ID})
RUN adduser --quiet --disabled-password --system --no-create-home --uid ${USER_ID} --gid ${GROUP_ID} --gecos '' --shell /bin/bash ${USER_NAME}
RUN usermod -d ${HOME_DIR} ${USER_NAME}
RUN adduser --quiet ${USER_NAME} sudo ; echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /data
WORKDIR /data

USER ${USER_NAME}
