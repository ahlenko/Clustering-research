"""Interactive experiment workflow used by the desktop interface.

The service deliberately imports optional models only when the user selects
them.  A missing optional package therefore produces a useful error for that
experiment instead of making the whole application unusable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import json
import os
from html import escape

import numpy as np
import pandas as pd

# Matplotlib otherwise attempts to create a cache in the user's home directory,
# which may be read-only on lab machines and in packaged environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/text_clustering_matplotlib")

from benchmark.benchmark_runner import BenchmarkRunner
from core.dataset import TextDataset
from metrics.external import evaluate_external
from metrics.internal import evaluate_internal
from preprocessing.pipeline import PreprocessingPipeline


Progress = Callable[[str], None]


@dataclass
class ExperimentOptions:
    input_path: str
    text_column: str = "text"
    label_column: str = "label"
    max_documents: int | None = None
    remove_stopwords: bool = True
    lemmatize: bool = True
    custom_stopwords: tuple[str, ...] = ()
    vectorizers: tuple[str, ...] = ("tfidf",)
    algorithms: tuple[str, ...] = ("kmeans",)
    n_clusters: int = 3
    kmeans_init: str = "k-means++"
    linkage: str = "ward"
    eps: float = 0.5
    min_samples: int = 5
    parallel_restarts: int = 8
    parallel_workers: int = 4
    runs: int = 1
    make_charts: bool = True
    recommend_clusters: bool = True


class ExperimentService:
    """Runs combinations of vectorization and clustering methods."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_output_dir = self.output_dir

    @staticmethod
    def available_vectorizers() -> tuple[str, ...]:
        return ("tfidf", "word2vec", "fasttext", "doc2vec", "sentence_bert")

    @staticmethod
    def available_algorithms() -> tuple[str, ...]:
        return ("kmeans", "parallel_kmeans", "hierarchical", "dbscan", "minibatch", "birch", "spectral", "affinity", "hdbscan")

    def run(self, options: ExperimentOptions, progress: Optional[Progress] = None) -> tuple[pd.DataFrame, dict]:
        say = progress or (lambda _message: None)
        if not options.vectorizers or not options.algorithms:
            raise ValueError("Оберіть щонайменше один метод векторизації та кластеризації")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_dir = self.output_dir / "experiments" / f"experiment_{stamp}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        dataset = TextDataset(options.input_path, options.text_column,
                              options.label_column or None, options.max_documents).load()
        if len(dataset.texts) < 2:
            raise ValueError("Для кластеризації потрібно щонайменше два документи")
        say(f"Завантажено документів: {len(dataset.texts)}")
        pipeline = PreprocessingPipeline(remove_stopwords=options.remove_stopwords,
                                         lemmatize=options.lemmatize,
                                         custom_stopwords=set(options.custom_stopwords))
        tokens = [pipeline.process(text) for text in dataset.texts]
        if not any(tokens):
            raise ValueError("Після очищення не залишилося слів. Змініть параметри обробки.")

        rows: list[dict] = []
        artifacts: dict[str, str] = {}
        benchmark = BenchmarkRunner(runs=max(1, options.runs))
        for vec_name in options.vectorizers:
            say(f"Векторизація: {vec_name}")
            vectorizer = self._vectorizer(vec_name)
            X, vector_stats = benchmark.run(vectorizer.fit_transform, tokens)
            X = np.asarray(X)
            recommended_k = self._recommend_k(X) if options.recommend_clusters else None
            for alg_name in options.algorithms:
                say(f"Кластеризація: {vec_name} + {alg_name}")
                try:
                    labels, cluster_stats = self._cluster(X, alg_name, options, benchmark)
                    labels = np.asarray(labels)
                    row = {
                        "dataset": Path(options.input_path).stem,
                        "vectorizer": vec_name,
                        "algorithm": alg_name,
                        "documents": len(dataset.texts),
                        "clusters_found": len(set(labels)) - (1 if -1 in labels else 0),
                        "noise_documents": int(np.sum(labels == -1)),
                        "vectorization_seconds": vector_stats["time_mean"],
                        "vectorization_memory_kb": vector_stats["memory_peak_mean"],
                        "clustering_seconds": cluster_stats["time_mean"],
                        "clustering_memory_kb": cluster_stats["memory_peak_mean"],
                        "cpu_percent": cluster_stats.get("cpu_mean"),
                        "cpu_per_core_percent": cluster_stats.get("cpu_per_core_mean"),
                        "recommended_k_silhouette": recommended_k,
                    }
                    row.update(evaluate_internal(X, labels))
                    if dataset.labels is not None:
                        row.update(evaluate_external(dataset.labels, labels))
                    if options.make_charts:
                        path = self._cluster_chart(X, labels, f"{vec_name} + {alg_name}")
                        artifacts[f"chart_{vec_name}_{alg_name}"] = str(path)
                        if alg_name == "hierarchical":
                            artifacts[f"dendrogram_{vec_name}"] = str(self._dendrogram(X, vec_name))
                    rows.append(row)
                except Exception as exc:  # A failed optional method should not hide other results.
                    rows.append({"dataset": Path(options.input_path).stem, "vectorizer": vec_name,
                                 "algorithm": alg_name, "error": str(exc)})
                    say(f"Не вдалося виконати {vec_name} + {alg_name}: {exc}")

        results = pd.DataFrame(rows)
        conclusion = self.conclusion(results)
        paths = self.export(results, stamp, conclusion)
        artifacts.update(paths)
        artifacts["conclusion"] = conclusion
        if "clustering_seconds" in results.columns and not results.dropna(
                subset=["clustering_seconds"]).empty:
            artifacts["comparison_chart"] = str(self._comparison_chart(results, stamp))
        artifacts["configuration"] = str(self._save_configuration(options, stamp))
        manifest_path = self.run_output_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "documents": len(dataset.texts),
            "artifacts": artifacts,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["manifest"] = str(manifest_path)
        return results, artifacts

    @staticmethod
    def load_saved_experiment(folder: str | Path) -> tuple[pd.DataFrame, dict]:
        """Load a completed experiment without running algorithms again."""
        folder = Path(folder)
        result_files = sorted(folder.glob("results_*.json"))
        if not result_files:
            raise FileNotFoundError("У вибраній папці не знайдено файл результатів JSON")
        results = pd.read_json(result_files[-1])
        manifest = folder / "manifest.json"
        artifacts: dict = {}
        if manifest.exists():
            artifacts = json.loads(manifest.read_text(encoding="utf-8")).get("artifacts", {})
        if not artifacts:
            artifacts = {"json": str(result_files[-1])}
        return results, artifacts

    def _vectorizer(self, name: str):
        if name == "tfidf":
            from vectorizers.tfidf import TfidfVectorizerAdapter
            return TfidfVectorizerAdapter()
        if name == "word2vec":
            from vectorizers.word2vec import Word2VecVectorizer
            return Word2VecVectorizer(min_count=1, workers=1)
        if name == "fasttext":
            from vectorizers.fasttext import FastTextVectorizer
            return FastTextVectorizer(min_count=1, workers=1)
        if name == "doc2vec":
            from vectorizers.doc2vec import Doc2VecVectorizer
            return Doc2VecVectorizer(min_count=1, workers=1)
        if name == "sentence_bert":
            from vectorizers.sentence_bert import SentenceBERTVectorizer
            return SentenceBERTVectorizer()
        raise ValueError(f"Невідомий векторизатор: {name}")

    def _cluster(self, X: np.ndarray, name: str, options: ExperimentOptions,
                 benchmark: BenchmarkRunner) -> tuple[np.ndarray, dict]:
        if name in {"kmeans", "parallel_kmeans", "hierarchical", "minibatch", "birch", "spectral"} and options.n_clusters > len(X):
            raise ValueError("Кількість кластерів не може перевищувати кількість документів")
        if name == "kmeans":
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=options.n_clusters, init=options.kmeans_init,
                           n_init="auto", random_state=42)
        elif name == "parallel_kmeans":
            from clustering.parallel_kmeans import ParallelKMeansClusterer
            model = ParallelKMeansClusterer(
                n_clusters=options.n_clusters,
                restarts=options.parallel_restarts,
                workers=options.parallel_workers,
                init=options.kmeans_init,
            )
        elif name == "hierarchical":
            from sklearn.cluster import AgglomerativeClustering
            # Ward linkage is defined only for Euclidean distance.
            model = AgglomerativeClustering(n_clusters=options.n_clusters, linkage=options.linkage)
        elif name == "dbscan":
            from sklearn.cluster import DBSCAN
            model = DBSCAN(eps=options.eps, min_samples=options.min_samples)
        elif name == "minibatch":
            from sklearn.cluster import MiniBatchKMeans
            model = MiniBatchKMeans(n_clusters=options.n_clusters, n_init="auto", random_state=42)
        elif name == "birch":
            from sklearn.cluster import Birch
            model = Birch(n_clusters=options.n_clusters)
        elif name == "spectral":
            from sklearn.cluster import SpectralClustering
            neighbors = max(1, min(10, len(X) - 1))
            model = SpectralClustering(n_clusters=options.n_clusters, n_neighbors=neighbors, random_state=42)
        elif name == "affinity":
            from sklearn.cluster import AffinityPropagation
            model = AffinityPropagation(random_state=42)
        elif name == "hdbscan":
            import hdbscan
            model = hdbscan.HDBSCAN(min_cluster_size=max(2, options.min_samples))
        else:
            raise ValueError(f"Невідомий алгоритм: {name}")
        return benchmark.run(model.fit_predict, X)

    @staticmethod
    def conclusion(results: pd.DataFrame) -> str:
        """Produce a concise, evidence-based conclusion for an experiment."""
        if "clustering_seconds" not in results.columns:
            return "Успішних запусків для формування висновку немає. Перевірте параметри та залежності."
        successful = results.dropna(subset=["clustering_seconds"])
        if successful.empty:
            return "Успішних запусків для формування висновку немає. Перевірте параметри та залежності."
        fastest = successful.loc[successful["clustering_seconds"].idxmin()]
        best = successful.loc[successful["silhouette"].idxmax()] if "silhouette" in successful else fastest
        parts = [
            f"Швидкодія: найшвидше виконався {fastest['vectorizer']} + {fastest['algorithm']} "
            f"за {fastest['clustering_seconds']:.4f} с.",
            f"Якість кластеризації: найкращий показник Silhouette має {best['vectorizer']} + "
            f"{best['algorithm']} ({best.get('silhouette', float('nan')):.4f}).",
        ]
        if "f1_macro" in successful and successful["f1_macro"].notna().any():
            external_best = successful.loc[successful["f1_macro"].idxmax()]
            parts.append(f"За еталонними мітками: найвищий F1 macro має {external_best['vectorizer']} + "
                         f"{external_best['algorithm']} ({external_best['f1_macro']:.4f}).")
        return "\n\n".join(parts)

    def export(self, results: pd.DataFrame, stamp: str, conclusion: str = "") -> dict[str, str]:
        csv_path = self.run_output_dir / f"results_{stamp}.csv"
        json_path = self.run_output_dir / f"results_{stamp}.json"
        html_path = self.run_output_dir / f"report_{stamp}.html"
        results.to_csv(csv_path, index=False)
        results.to_json(json_path, orient="records", indent=2, force_ascii=False)
        conclusion_html = "</p><p>".join(escape(part) for part in conclusion.split("\n\n"))
        html_path.write_text("<meta charset='utf-8'><h1>Text clustering report</h1><h2>Conclusion</h2><p>" +
                             conclusion_html + "</p><h2>Results</h2>" +
                             results.to_html(index=False, float_format=lambda x: f"{x:.4f}"), encoding="utf-8")
        return {"csv": str(csv_path), "json": str(json_path), "html": str(html_path)}

    def _cluster_chart(self, X: np.ndarray, labels: np.ndarray, title: str) -> Path:
        from visualization.plots import plot_clusters
        name = "clusters_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
        path = self.run_output_dir / name
        # PCA is reliable for small corpora; t-SNE requires a larger sample.
        plot_clusters(X, labels, title=title, save_path=str(path), method="pca")
        return path

    def _comparison_chart(self, results: pd.DataFrame, stamp: str) -> Path:
        # This function is called by the background experiment worker.  The
        # report needs a PNG file, not a second GUI event loop.
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        successful = results.dropna(subset=["clustering_seconds"])
        path = self.run_output_dir / f"comparison_{stamp}.png"
        labels = successful["vectorizer"] + " + " + successful["algorithm"]
        plt.figure(figsize=(max(8, len(successful) * 1.1), 5))
        plt.bar(labels, successful["clustering_seconds"], color="#2f6f9f")
        plt.ylabel("Час кластеризації, с")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    @staticmethod
    def _recommend_k(X: np.ndarray) -> int | None:
        """Return the K with the best silhouette score for a compact range."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        upper = min(10, len(X) - 1)
        if upper < 2:
            return None
        best_k, best_score = None, float("-inf")
        for k in range(2, upper + 1):
            labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    def _dendrogram(self, X: np.ndarray, vectorizer: str) -> Path:
        """Save the hierarchical structure independently of the selected cut."""
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        path = self.run_output_dir / f"dendrogram_{vectorizer}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        plt.figure(figsize=(9, 5))
        dendrogram(linkage(X, method="ward"), no_labels=len(X) > 40, color_threshold=None)
        plt.title(f"Дендрограма: {vectorizer}")
        plt.ylabel("Відстань")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def _save_configuration(self, options: ExperimentOptions, stamp: str) -> Path:
        path = self.run_output_dir / f"configuration_{stamp}.json"
        path.write_text(json.dumps(asdict(options), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
