# How this port differs

The train judge's image leaves out the final suite's manifest, frames and public data, but not the
tensors that hold every episode. A value or a line of the judge also differs: 1024 commands on the
agent's GPU runners, not 32.
