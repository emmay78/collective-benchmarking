import sys

# NCCL strings for functions, algorithms, and protocols
FUNC = ["Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce"]
ALGO = ["Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree"]
PROTO = ["LL", "LL128", "Simple"]

# String Lengths
FUNC_NAME_LEN = 15
LATENCY_LEN = 8
BANDWIDTH_LEN = 7
ENTRY_LEN = LATENCY_LEN + BANDWIDTH_LEN + 2
ALGO_ENTRY_LEN = ENTRY_LEN * 3

# Import tuning table
data = open(sys.argv[1], "r")
tuning_table = data.read().splitlines()
data.close()

del tuning_table[0:3]
del tuning_table[len(FUNC):len(FUNC)+3]

# Parse tuning table
latencies = {}
bandwidths = {}

for block in range(2):
    for ba in range(len(ALGO)//2):
        a = ba + block
        for p in range(len(PROTO)):
            for f in range(len(FUNC)):
                algo = ALGO[a]
                proto = PROTO[p]
                func = FUNC[f]

                string_pos = FUNC_NAME_LEN + ALGO_ENTRY_LEN*ba + ENTRY_LEN*p
                latency = tuning_table[f + block*len(FUNC)][string_pos:string_pos+LATENCY_LEN]
                bandwidth = tuning_table[f + block*len(FUNC)][string_pos+LATENCY_LEN+1:string_pos+LATENCY_LEN+BANDWIDTH_LEN+1]

                latencies[tuple([algo, proto, func])] = float(latency)
                bandwidths[tuple([algo, proto, func])] = float(bandwidth)

def calculate_all_reduce_time(data_size, latencies, bandwidths):
    data_size /= 1000 # convert data_size from MB to GB
    min_time = float('inf')
    for proto in PROTO:
        latency = latencies[tuple(["Tree", proto, "AllReduce"])] * 1e6 # latency in us
        bandwidth = bandwidths[tuple(["Tree", proto, "AllReduce"])]    # BW in GB/s

        # break if no data
        if bandwidth == 0 or latency == 0:
            break

        time = latency + data_size/bandwidth
        min_time = min(time, min_time)

    return min_time


print(f"{calculate_all_reduce_time(float(sys.argv[2]), latencies, bandwidths):0.3f} seconds for {float(sys.argv[2])} MB")