from .padding import Padding
from .slice_layer import Slice_Layer

# When loading models with: "model.load('file.h5', custom_objects=custom_objects)"
custom_objects = {
    "Padding": Padding,
    "Slice_Layer": Slice_Layer
}