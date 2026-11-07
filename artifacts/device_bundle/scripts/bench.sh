#!/system/bin/sh
# bench.sh — run a command, measure wall time (ns) and peak RSS (VmHWM).
# usage: bench.sh <stdout_file> <cmd> [args...]
# NOTE: mksh arithmetic is 32-bit; use awk (double) for the ns->ms conversion.
out="$1"; shift
start=$(date +%s%N)
"$@" > "$out" 2>&1 &
pid=$!
max=0
while kill -0 "$pid" 2>/dev/null; do
  v=$(grep VmHWM /proc/$pid/status 2>/dev/null | awk '{print $2}')
  if [ -n "$v" ] && [ "$v" -gt "$max" ]; then max=$v; fi
  sleep 0.05
done
wait "$pid"
rc=$?
end=$(date +%s%N)
ms=$(awk "BEGIN{printf \"%d\", ($end-$start)/1000000}")
echo "WALL_MS=$ms VmHWM_KB=$max RC=$rc"
