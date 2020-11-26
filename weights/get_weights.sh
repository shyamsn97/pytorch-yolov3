#!/bin/bash

base_url="https://pjreddie.com/media/files"
for model in yolov3-tiny; do
  wget -P models/ "$base_url/$model.weights"
done
