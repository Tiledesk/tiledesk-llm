#!/bin/bash

# Parse command-line arguments
myargs=$(getopt --name "$0" -o hr --long "help,environment:": -- "$@" )

eval set -- "$myargs"

while true;
do
  case "$1" in
    -h|--help)
      echo "usage $0 --help --environment"
      shift
      ;;

    -e|--environment)
      shift 2
      environment="$2"
      break
      ;;

    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    --)
      shift
      break
      ;;
  esac
done
echo "$environment"
# Validate required argument
if [ -z "$environment" ]; then
  echo "Info: --environment argument is set to dev."
  environment="dev"

fi

echo "$environment"
# Connect to Redis using the parsed URL
##tilellm --redis_url "$redisurl"
gunicorn --bind 0.0.0.0:8000  -w 2 --env ENVIRON="$environment" --worker-class uvicorn.workers.UvicornWorker tilellm.__main__:app

