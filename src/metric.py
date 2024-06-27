import math
from pathlib import Path

from scipy.stats import pearsonr

from numpy import dot
from numpy.linalg import norm

from embed import EMBEDDING
from util import from_json, read_from_file


def distance(a: list[float] = None, b: list[float] = None) -> float:
    assert a is not None
    assert b is not None
    assert len(a) == len(b)
    return math.dist(a, b)


def cosine_similarity(a: list[float] = None, b: list[float] = None) -> float:
    assert a is not None
    assert b is not None
    assert len(a) == len(b)
    return dot(a, b) / (norm(a) * norm(b))


def correlation(a: list[float] = None, b: list[float] = None) -> float:
    assert a is not None and len(a) > 0
    assert b is not None and len(b) > 0
    assert len(a) == len(b), str(len(a)) + "<>" + str(len(b))
    correl, _ = pearsonr(a, b)
    return correl


def create_metrics(dataset: Path = None) -> dict:
    ds_dict = from_json(read_from_file(dataset))
    metrics: dict = {"cosine_similarity": {}, "distance": {}}
    for item in ds_dict.values():
        embedding = item[EMBEDDING]
        for engine in embedding:
            if not isinstance(embedding[engine], dict):
                if engine not in metrics:
                    metrics[engine] = []
                metrics[engine].append(embedding[engine])
                continue
            if engine not in metrics["cosine_similarity"]:
                metrics["cosine_similarity"][engine] = {}
            if engine not in metrics["distance"]:
                metrics["distance"][engine] = {}
            for key0 in embedding[engine].keys():
                for key1 in embedding[engine].keys():
                    if not key0 == key1:  # use alphabetical sort to avoid duplicates
                        if key0 >= key1:
                            key = key0 + "-" + key1
                        else:
                            continue  # skip duplicate
                        if key not in metrics["cosine_similarity"][engine]:
                            metrics["cosine_similarity"][engine][key] = []
                        if key not in metrics["distance"][engine]:
                            metrics["distance"][engine][key] = []
                        metrics["cosine_similarity"][engine][key].append(cosine_similarity(embedding[engine][key0],
                                                                                           embedding[engine][key1]))
                        metrics["distance"][engine][key].append(distance(embedding[engine][key0],
                                                                         embedding[engine][key1]))
    return metrics


def correlate_embeddings(dataset: Path = None) -> dict:
    metrics = create_metrics(dataset)
    eng_correlation: dict = {}
    for key in metrics:
        eng_correlation[key] = {}
    cleanup = []
    for metric1, engine_keys1 in eng_correlation.items():
        if isinstance(metrics[metric1], list):
            for metric2, engine_keys2 in eng_correlation.items():
                if metric2 != metric1:
                    if isinstance(metrics[metric2], list):
                        engine_keys1[metric1 + "<>" + metric2] = correlation(metrics[metric1], metrics[metric2])
                    else:
                        for engine0, _ in metrics[metric2].items():
                            key = next(iter(metrics[metric2][engine0].keys()))
                            engine_keys2[metric1 + "<>" + engine0] = (
                                correlation(metrics[metric1],
                                            metrics[metric2][engine0][key]))
            cleanup.append(metric1)
        else:
            for engine0, _ in metrics[metric1].items():
                for engine1, _ in metrics[metric1].items():
                    if engine0 > engine1:  # alphabetical order to avoid duplicates
                        engine_key = engine0 + "<>" + engine1
                        for key, _ in metrics[metric1][engine0].items():
                            engine_keys1[engine_key] = correlation(metrics[metric1][engine0][key],
                                                                   metrics[metric1][engine1][key])
    for metric in cleanup:
        del eng_correlation[metric]
    return eng_correlation
