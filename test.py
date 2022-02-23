import time

start = time.perf_counter()
for i in range(2):
	a = i*2
	time.sleep(1)
end = time.perf_counter()

print(end - start)