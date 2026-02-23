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
  environment=""
fi

# --- Configurazione Gunicorn ---
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

# --- Configurazione TaskIQ (variabili separate da Gunicorn) ---
TASKIQ_WORKER_COUNT=${TASKIQ_WORKERS:-2}       # Numero di worker taskiq (CPU cores - 1)
PREFETCH=${TASKIQ_PREFETCH:-1}                 # CRITICO: 1 task alla volta
TASKIQ_LOG_LEVEL=${TASKIQ_LOG_LEVEL:-INFO}     # 'debug' solo per troubleshooting
SHUTDOWN_TIMEOUT=${TASKIQ_SHUTDOWN_TIMEOUT:-30} # Tempo per shutdown graceful
MAX_ASYNC_TASKS=${TASKIQ_MAX_ASYNC_TASKS:-10}   # Task concorrenti per worker
TASKIQ_RESTART_DELAY=${TASKIQ_RESTART_DELAY:-5} # Secondi prima di riavviare il worker

SUPERVISOR_PID=""
GUNICORN_PID=""

# --- Funzione di cleanup per segnali ---
cleanup() {
    echo "Received shutdown signal, stopping services gracefully..."

    if [ -n "$SUPERVISOR_PID" ]; then
        kill -TERM "$SUPERVISOR_PID" 2>/dev/null || true
    fi

    if [ -n "$GUNICORN_PID" ]; then
        kill -TERM "$GUNICORN_PID" 2>/dev/null || true
    fi

    # Attende shutdown graceful
    sleep "$SHUTDOWN_TIMEOUT"

    # Forza chiusura se necessario
    kill -9 "$SUPERVISOR_PID" "$GUNICORN_PID" 2>/dev/null || true

    exit 0
}

# --- Registra handler per segnali ---
trap cleanup SIGTERM SIGINT SIGHUP

if [ "$ENABLE_TASKIQ" = "true" ]; then
    echo "Starting TaskIQ worker supervisor (workers=$TASKIQ_WORKER_COUNT)..."

    # Loop di supervisione: riavvia il worker se crasha
    (
        while true; do
            python -m taskiq worker \
                tilellm.modules.task_executor.broker:broker \
                tilellm.modules.task_executor.tasks \
                --workers "$TASKIQ_WORKER_COUNT" \
                --max-prefetch "$PREFETCH" \
                --max-async-tasks "$MAX_ASYNC_TASKS" \
                --shutdown-timeout "$SHUTDOWN_TIMEOUT" \
                --log-level "$TASKIQ_LOG_LEVEL"

            EXIT_CODE=$?
            # Exit code 0 = shutdown pulito (SIGTERM), non riavviare
            if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 143 ]; then
                echo "TaskIQ worker stopped cleanly (exit $EXIT_CODE)"
                break
            fi
            echo "TaskIQ worker crashed (exit $EXIT_CODE), restarting in ${TASKIQ_RESTART_DELAY}s..."
            sleep "$TASKIQ_RESTART_DELAY"
        done
    ) &

    SUPERVISOR_PID=$!
    echo "TaskIQ supervisor started (PID: $SUPERVISOR_PID)"
else
    echo "TaskIQ worker disabled (ENABLE_TASKIQ=$ENABLE_TASKIQ)"
fi

echo "Starting gunicorn with workers=$WORKERS timeout=$TIMEOUT max-requests=$MAXREQUESTS jitter=$MAXRJITTER graceful-timeout=$GRACEFULTIMEOUT"

python -m gunicorn \
          --bind 0.0.0.0:8000  \
          --workers $WORKERS \
          --timeout $TIMEOUT \
          --max-requests $MAXREQUESTS \
          --max-requests-jitter $MAXRJITTER \
          --graceful-timeout $GRACEFULTIMEOUT \
          --log-config-json log_conf.json \
          --worker-class uvicorn.workers.UvicornWorker \
          tilellm.__main__:app &

GUNICORN_PID=$!
echo "Gunicorn started (PID: $GUNICORN_PID)"

# --- Attende entrambi i processi ---
echo "Waiting for services..."
wait "$SUPERVISOR_PID" "$GUNICORN_PID" 2>/dev/null || true

echo "Services stopped"
