#!/bin/bash
# MaxK-GNN training wrapper with proper library path
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
exec "$@"
