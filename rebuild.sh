#!/bin/bash
sg docker -c "docker compose build web"
sg docker -c "docker compose up -d --no-build web"
