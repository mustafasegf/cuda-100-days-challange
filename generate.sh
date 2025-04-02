#!/usr/bin/env bash

last_day=$(<.lastday)
day=$((last_day + 1))

padded=$(printf "%03d" "$day")

mkdir -p "day-${padded}"

echo "$day" > .lastday

