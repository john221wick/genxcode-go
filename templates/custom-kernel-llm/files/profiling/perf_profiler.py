class PerfProfiler:
    @staticmethod
    def record_command(output="perf.data", pid=None, duration=10):
        if pid:
            return f"perf record -g -p {pid} -o {output} -- sleep {duration}"
        return f"perf record -g -o {output} -- python train.py"

    @staticmethod
    def stat_command(events=None):
        if events is None:
            events = "cache-misses,cache-references,instructions,cycles"
        return f"perf stat -e {events} python main.py train --config debug"

    @staticmethod
    def report_command(data_file="perf.data"):
        return f"perf report -i {data_file}"

    @staticmethod
    def flamegraph_command(data_file="perf.data", output="flame.svg"):
        return (
            f"perf script -i {data_file} | "
            f"stackcollapse-perf.pl | flamegraph.pl > {output}"
        )
