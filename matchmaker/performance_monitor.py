from timeit import default_timer

class PerformanceMonitor():
    def __init__(self):
        self.timings = {}
        self.current_times = {}

    def start_block(self,category:str):
        self.current_times[category] = default_timer()

    def stop_block(self,category:str,instances:int=1):
        if not category in self.timings:
            self.timings[category] = (0,0)

        time,old_instances = self.timings[category]

        self.timings[category] = (time + default_timer() - self.current_times[category], old_instances + instances)

    def print_summary(self):
        for cat,(time,instances) in self.timings.items():
            if instances > 1:
                print(cat, instances/time, "it/s")
            else:
                print(cat, time, "s")

    def save_summary(self, file):
        with open(file, "w") as out_file:
            for cat,(time,instances) in self.timings.items():
                if instances > 1:
                    out_file.write(cat + str(instances/time) + "it/s\n")
                else:
                    out_file.write(cat + str(time) + "s\n")