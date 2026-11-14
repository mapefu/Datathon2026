"""LightGBM → arrays planos para `agent.Forest` (evaluación con numpy, sin LightGBM en el servidor)."""

from __future__ import annotations

import numpy as np

MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def flatten(models: list[tuple[object, list[int]]]) -> dict[str, np.ndarray]:
    """models: [(booster, columnas globales de cada feature local)] en el orden de params["models"].

    Todos los nodos (internos y hojas) en un único array; cada hoja apunta a sí misma con umbral +inf."""
    feat, thr, dleft, mtype, left, right, value, leaf = [], [], [], [], [], [], [], []
    roots, tree_model = [], []

    def add(node, cols):
        i = len(feat)
        feat.append(0), thr.append(np.inf), dleft.append(True), mtype.append(0)
        left.append(i), right.append(i), value.append(0.0), leaf.append(True)
        if "split_feature" not in node:
            value[i] = float(node["leaf_value"])
            return i
        if node.get("decision_type", "<=") != "<=":
            raise ValueError(f"split no numérico: {node.get('decision_type')}")
        leaf[i] = False
        feat[i] = int(cols[node["split_feature"]])
        thr[i] = float(node["threshold"])
        dleft[i] = bool(node["default_left"])
        mtype[i] = MISSING[node["missing_type"]]
        left[i] = add(node["left_child"], cols)
        right[i] = add(node["right_child"], cols)
        return i

    for mi, (booster, cols) in enumerate(models):
        for tree in booster.dump_model()["tree_info"]:
            roots.append(add(tree["tree_structure"], cols))
            tree_model.append(mi)
    return {
        "F_feat": np.asarray(feat, np.int32), "F_thr": np.asarray(thr, np.float64),
        "F_dleft": np.asarray(dleft, np.int8), "F_mtype": np.asarray(mtype, np.int8),
        "F_left": np.asarray(left, np.int32), "F_right": np.asarray(right, np.int32),
        "F_value": np.asarray(value, np.float64), "F_leaf": np.asarray(leaf, np.int8),
        "F_roots": np.asarray(roots, np.int32), "F_tree_model": np.asarray(tree_model, np.int16),
    }
