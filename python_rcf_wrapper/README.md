# Random Cut Forest (RCF) in Python

RCF (Random Cut Forest) is implemented in Java and Rust. To use it in Python, follow these steps:

## Step 1: Install JPype

Install JPype to enable the interaction between Python and Java. You can find the installation instructions at [JPype Installation](https://jpype.readthedocs.io/en/latest/install.html).

## Step 2: Import and Use TRCF from `python_rcf_wrapper`

You need to import `TRCF` from the `python_rcf_wrapper` and call its `process` method. Below is an example Python script to demonstrate this:

```python
from python_rcf_wrapper.trcf_model import TRandomCutForestModel as TRCF
import numpy as np

# Parameters for the RCF model
shingle_size = 8
dimensions = 2
num_trees = 50
output_after = 32
sample_size = 256

# Initialize the RCF model
model = TRCF(
    rcf_dimensions=shingle_size * dimensions,
    shingle_size=shingle_size,
    num_trees=num_trees,
    output_after=output_after,
    anomaly_rate=0.001,
    z_factor=3,
    score_differencing=0.5,
    sample_size=sample_size
)

# Generate test data
TEST_DATA = np.random.normal(size=(300, 2))

# Process each data point and print the RCF score and anomaly grade
for point in TEST_DATA:
    descriptor = model.process(point)
    print("RCF score: {}, Anomaly grade: {}".format(descriptor.getRCFScore(), descriptor.getAnomalyGrade()))
```
