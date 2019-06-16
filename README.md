# MLModelScope Caffe Agent

[![Build Status](https://travis-ci.org/rai-project/caffe.svg?branch=master)](https://travis-ci.org/rai-project/caffe)
[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/caffe)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=15)
[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/caffe)](https://goreportcard.com/report/github.com/rai-project/caffe)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/caffe:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/caffe:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/caffe:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/caffe:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/caffe:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/caffe:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/caffe:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/caffe:amd64-gpu-latest 'Get your own version badge on microbadger.com')

## Installation

Install go if you have not done so. Please follow [Go Installation](https://docs.mlmodelscope.org/installation/source/golang).

Download and install the MLModelScope Caffe Agent:

```
go get -v github.com/rai-project/caffe

```

The agent requires The Caffe C library and other Go packages.

### Go packages

You can install the dependency through `go get`.

```
cd $GOPATH/src/github.com/rai-project/caffe
go get -u -v ./...
```

Or use [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`.

Note: The CGO interface passes go pointers to the C API. This is an error by the CGO runtime. Disable the error by placing

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`


### The Caffe C library

The Caffe C library is required.

If you use Caffe Docker Images (e.g. NVIDIA GPU CLOUD (NGC)), skip this step.

Refer to [go-caffe](https://github.com/rai-project/go-caffe#caffe-installation) for caffe installation.


## External services

Refer to [External services](https://github.com/rai-project/tensorflow#external-services).

## Use within Caffe Docker Images

Refer to [Use within TensorFlow Docker Images](https://github.com/rai-project/tensorflow#use-within-tensorflow-docker-images).

## Usage

Refer to [Usage](https://github.com/rai-project/tensorflow#usage)
