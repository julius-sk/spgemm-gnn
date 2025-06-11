import dgl.nn as dglnn
import inspect

# Print out the methods of SAGEConv
methods = [method for method in dir(dglnn.SAGEConv) if not method.startswith('_')]
print(f"Available methods in SAGEConv: {methods}")

# Let's look at the actual implementation
source = inspect.getsource(dglnn.SAGEConv.forward)
print("\nSAGEConv forward method source:")
print(source)