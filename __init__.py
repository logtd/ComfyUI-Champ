from .nodes.apply_champ_features_node import ApplyChampFeaturesNode
from .nodes.load_champ_guider_node import LoadChampGuiderNode
from .nodes.get_champ_feature_node import GetChampFeatureNode


NODE_CLASS_MAPPINGS = {
    "ApplyChampFeatures": ApplyChampFeaturesNode,
    "LoadChampGuider": LoadChampGuiderNode,
    "GetChampFeature": GetChampFeatureNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyChampFeatures": "Apply Champ Features",
    "LoadChampGuider": "Load Champ Guider",
    "GetChampFeature": "Get Champ Feature",
}
