FROM ubuntu:18.04
MAINTAINER Michael Spieler (nuft) <michael.spieler@epfl.ch>
LABEL Description="Image for building PolyMPC project"

WORKDIR /work
ADD . /work

# Update and install dependencies
RUN apt-get update && \
    apt-get upgrade -y

RUN bash install.sh
