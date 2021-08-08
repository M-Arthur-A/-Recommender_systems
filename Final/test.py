from src.utils import *
globals().update(load_settings())

data = Dataset()
data.prefilter_items()