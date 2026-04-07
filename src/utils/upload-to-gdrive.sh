#!/bin/zsh

cd /Users/canderson/Documents/wedding || echo Err: dir not found
rclone --progress copyto amazon-registry  gdrive:~/wedding/amazon-registry