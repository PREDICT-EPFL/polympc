# Docker

## Create a Docker image

You can create the docker image from the Dockerfile in this directory:
```bash
cd polympc/ci/docker
docker build -t polympc-dev .
```

## Build and run locally using Docker

Run a new container form the Docker image.

```bash
cd polympc
docker run -it -v (pwd):/work/polympc polympc-dev bash
```

This gives a shell with the working directory mounted under `/work/polympc`.

```bash
cd polympc
mkdir -p build && cd build
cmake ..
make
```

To run tests from a separate shell, find the running containter id using `docker ps` and then start a new bash session.

```bash
docker exec -it <CONTAINTER_ID> bash
```
