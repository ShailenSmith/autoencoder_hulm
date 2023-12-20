from matplotlib import pyplot as plt
import tensorboard as tb
from packaging import version
from tensorflow.python.summary.summary_iterator import summary_iterator


major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
        "This notebook requires TensorBoard 2.3 or later."
print("Tensorboard version: ", tb.__version__)

print("\n\n\n")


path_to_events_file = "/cronus_data/ssmith/models/blogsUD/sepsep_model/runs/sepsep_12-12/events.out.tfevents.1702423751.cronus.3679986.1"


count = 0
for e in summary_iterator(path_to_events_file):
    for v in e.summary.value:
        if v.tag == 'loss':
            print(v.simple_value)
            count += 1
        if count == 5:
            exit()
