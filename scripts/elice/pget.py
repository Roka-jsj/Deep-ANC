#!/usr/bin/env python3
"""병렬 범위(Range) 다운로더 — 연결당 속도 제한 우회. 사용: pget.py URL OUT [N]"""
import sys
import threading
import time
import urllib.request

url, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 16

req = urllib.request.Request(url, method="HEAD")
total = int(urllib.request.urlopen(req).headers["Content-Length"])
print(f"total {total/1e9:.2f} GB, {n} connections", flush=True)

with open(out, "wb") as f:
    f.truncate(total)

done = [0] * n


def worker(i):
    lo = total * i // n
    hi = total * (i + 1) // n - 1
    for attempt in range(8):
        if lo + done[i] > hi:
            return
        try:
            r = urllib.request.Request(url, headers={"Range": f"bytes={lo+done[i]}-{hi}"})
            with urllib.request.urlopen(r, timeout=60) as resp, open(out, "r+b") as f:
                f.seek(lo + done[i])
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        return
                    f.write(chunk)
                    done[i] += len(chunk)
        except Exception as e:
            print(f"[{i}] retry {attempt}: {e}", flush=True)
            time.sleep(2)


threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
t0 = time.time()
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    time.sleep(5)
    got = sum(done)
    print(f"{got/1e9:.2f}/{total/1e9:.2f} GB  {got/1e6/max(1,time.time()-t0):.1f} MB/s", flush=True)
print("DONE", flush=True)
