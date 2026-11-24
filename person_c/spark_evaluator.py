"""Person C: Spark-based Parallel Population Fitness Evaluator.

Agreed interface:
    evaluate_population(population, fitness_fn) -> list[float]
"""
import os
from typing import Callable, List, Tuple
import numpy as np

# Type aliases matching shared/interfaces.py
Population = List[Tuple[np.ndarray, int]]
FitnessFn = Callable[[np.ndarray, int], float]

_SPARK_SESSION = None


def get_spark_session(app_name: str = "QIEA_Fitness_Evaluator", master: str = "local[*]"):
    """Initialize or reuse a lightweight local PySpark Session."""
    global _SPARK_SESSION
    if _SPARK_SESSION is None:
        try:
            from pyspark.sql import SparkSession
            _SPARK_SESSION = (
                SparkSession.builder
                .appName(app_name)
                .master(master)
                .config("spark.driver.memory", "2g")
                .config("spark.ui.showConsoleProgress", "false")
                .getOrCreate()
            )
        except Exception:
            _SPARK_SESSION = False
    return _SPARK_SESSION if _SPARK_SESSION is not False else None


def evaluate_population(population: Population, fitness_fn: FitnessFn) -> List[float]:
    """Parallel evaluation of population fitness across Spark partitions.
    
    If Spark is unavailable or fails to initialize, it seamlessly falls back to 
    the serial contract evaluator so runs never crash.
    """
    spark = get_spark_session()

    if spark is None:
        # Fallback to serial evaluation
        return [float(fitness_fn(mask, k)) for mask, k in population]

    try:
        sc = spark.sparkContext
        # Serialize numpy masks into booleans for safe worker pickling
        serializable_pop = [(mask.tolist(), int(k)) for mask, k in population]
        
        # Parallelize across RDD partitions
        rdd = sc.parallelize(serializable_pop, numSlices=min(len(population), os.cpu_count() or 4))
        
        # Map worker task
        def _eval_worker(item):
            mask_list, k = item
            mask_arr = np.array(mask_list, dtype=bool)
            return float(fitness_fn(mask_arr, k))

        scores = rdd.map(_eval_worker).collect()
        return [float(s) for s in scores]
    except Exception:
        # Fallback if Spark worker serialization has issues
        return [float(fitness_fn(mask, k)) for mask, k in population]