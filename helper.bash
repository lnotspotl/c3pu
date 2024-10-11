#!/usr/bin/env bash

function print_help() {
	echo "Usage: ./tbai [--format|--lint]"
}

if [[ "$1" == "--format" ]]; then
    ruff format .
    exit $?
fi

if [[ "$1" == "--lint" ]]; then
    ruff lint .
    exit $?
fi


print_help