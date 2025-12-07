#!/usr/bin/env bash
# Simple entrypoint: if no args, sleep forever to keep container alive for docker-compose service overrides.
if [ "$#" -eq 0 ]; then
  # Run a sleep loop to keep the container alive
  while true; do sleep 1000; done
else
  # Execute provided command
  exec "$@"
fi

