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
# Validate required argument
if [ -z "$environment" ]; then
  # echo "Info: --environment argument is set to dev."
  environment=""

fi
if [ -z "$WORKERS" ]; then
  WORKERS=3
fi
if [ -z "$TIMEOUT" ]; then
  TIMEOUT=240
fi
if [ -z "$MAXREQUESTS" ]; then
  MAXREQUESTS=2000
fi
if [ -z "$MAXRJITTER" ]; then
  MAXRJITTER=5
fi
if [ -z "$GRACEFULTIMEOUT" ]; then
  GRACEFULTIMEOUT=60
fi

if [ "$ENABLE_TASKIQ" = "true" ]; then
    echo "Starting taskiq worker..."
    python -m taskiq worker tilellm.modules.task_executor.broker:broker tilellm.modules.task_executor.tasks &
fi

echo "start gunicorn with workers $WORKERS --timeout $TIMEOUT --max-requests $MAXREQUESTS --max-requests-jitter $MAXRJITTER --graceful-timeout $GRACEFULTIMEOUT"

python -m gunicorn --bind 0.0.0.0:8000  --workers $WORKERS --timeout $TIMEOUT --max-requests $MAXREQUESTS --max-requests-jitter $MAXRJITTER --graceful-timeout $GRACEFULTIMEOUT --log-config-json log_conf.json --worker-class uvicorn.workers.UvicornWorker tilellm.__main__:app

