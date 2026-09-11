"""A small Tk desktop interface for text-clustering experiments."""
from __future__ import annotations

import queue
import threading
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core.experiment_service import ExperimentOptions, ExperimentService
from preprocessing.stopwords import StopwordRemover


class CollapsibleSection(ttk.Frame):
    """A titled section whose contents can be hidden to save vertical space."""
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.title = title
        self.expanded = True
        self.columnconfigure(0, weight=1)
        self.toggle_button = ttk.Button(self, command=self.toggle)
        self.toggle_button.grid(row=0, column=0, sticky="ew")
        # The header already supplies a title; an empty LabelFrame reserves an
        # unnecessary title gap above its contents.
        self.content = ttk.Frame(self, padding=(6, 2, 6, 6))
        self.content.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)
        self._set_title()

    def _set_title(self) -> None:
        marker = "▾" if self.expanded else "▸"
        self.toggle_button.configure(text=f"{marker}  {self.title}")

    def toggle(self) -> None:
        self.expanded = not self.expanded
        if self.expanded:
            self.content.grid()
        else:
            self.content.grid_remove()
        self._set_title()

    def collapse(self) -> None:
        if self.expanded:
            self.toggle()


class ClusteringApp(ttk.Frame):
    def __init__(self, root: tk.Tk):
        super().__init__(root, padding=14)
        self.root = root
        self.service = ExperimentService()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.row_chart_paths: dict[str, str] = {}
        self.row_cpu_details: dict[str, object] = {}
        self._hovered_item = ""
        self._graph_tip: tk.Toplevel | None = None
        self._tip_image: tk.PhotoImage | None = None
        self._build()
        self.pack(fill="both", expand=True)

    def _build(self) -> None:
        self.root.title("Дослідження кластеризації текстів")
        self.root.minsize(960, 650)
        ttk.Style().configure("Conclusion.TLabel", foreground="#155E75", font=("TkDefaultFont", 10, "bold"))
        self.columnconfigure(0, weight=1)
        # Method selectors and the results table both receive extra vertical
        # space when the user enlarges the window.
        self.rowconfigure(1, weight=1)
        self.rowconfigure(3, weight=3)
        self.data_section = CollapsibleSection(self, "Дані")
        self.data_section.grid(row=0, column=0, sticky="ew")
        source = self.data_section.content
        source.columnconfigure(1, weight=1)
        self.path = tk.StringVar()
        self.text_column = tk.StringVar(value="text")
        self.label_column = tk.StringVar(value="label")
        self.max_documents = tk.StringVar()
        ttk.Label(source, text="Файл TXT, CSV або JSON:").grid(row=0, column=0, sticky="w")
        ttk.Entry(source, textvariable=self.path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(source, text="Обрати…", command=self._choose_file).grid(row=0, column=2)
        ttk.Label(source, text="Колонка тексту:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source, textvariable=self.text_column, width=20).grid(row=1, column=1, sticky="w", padx=6, pady=(8, 0))
        ttk.Label(source, text="Колонка еталонних міток (необов’язково):").grid(row=1, column=1, sticky="e", pady=(8, 0))
        ttk.Entry(source, textvariable=self.label_column, width=20).grid(row=1, column=2, pady=(8, 0))
        ttk.Label(source, text="Макс. кількість текстів (порожньо = усі):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source, textvariable=self.max_documents, width=12).grid(row=2, column=1, sticky="w", padx=6, pady=(8, 0))

        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        controls.columnconfigure(0, weight=2)
        controls.columnconfigure(1, weight=1)
        controls.rowconfigure(0, weight=1)
        self.methods_section = CollapsibleSection(controls, "Методи")
        self.methods_section.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        methods = self.methods_section.content
        methods.columnconfigure(0, weight=1)
        methods.columnconfigure(1, weight=1)
        methods.rowconfigure(1, weight=1)
        ttk.Label(methods, text="Векторизація (можна кілька):").grid(row=0, column=0, sticky="w")
        ttk.Label(methods, text="Кластеризація (можна кілька):").grid(row=0, column=1, sticky="w")
        self.vector_list = tk.Listbox(methods, selectmode="extended", exportselection=False)
        self.algorithm_list = tk.Listbox(methods, selectmode="extended", exportselection=False)
        for method in self.service.available_vectorizers(): self.vector_list.insert("end", method)
        for method in self.service.available_algorithms(): self.algorithm_list.insert("end", method)
        self.vector_list.selection_set(0)
        self.algorithm_list.selection_set(0, 2)  # required baseline algorithms
        self.vector_list.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.algorithm_list.grid(row=1, column=1, sticky="nsew")
        self.parameters_section = CollapsibleSection(controls, "Параметри")
        self.parameters_section.grid(row=0, column=1, sticky="nsew")
        params = self.parameters_section.content
        self.n_clusters = tk.StringVar(value="3")
        self.eps = tk.StringVar(value="0.5")
        self.min_samples = tk.StringVar(value="5")
        self.parallel_restarts = tk.StringVar(value="8")
        self.parallel_workers = tk.StringVar(value="4")
        self.runs = tk.StringVar(value="1")
        self.custom_stopwords = tk.StringVar()
        self.stopword_count = tk.StringVar()
        self.kmeans_init = tk.StringVar(value="k-means++")
        self.linkage = tk.StringVar(value="ward")
        self.stopwords = tk.BooleanVar(value=True)
        self.lemmatize = tk.BooleanVar(value=True)
        self.charts = tk.BooleanVar(value=True)
        self.recommend_k = tk.BooleanVar(value=True)
        for column in (1, 3):
            params.columnconfigure(column, weight=1)
        self._field(params, 0, "Кількість кластерів:", self.n_clusters, column=0)
        self._field(params, 0, "Ініціалізація K-Means:", self.kmeans_init, ("k-means++", "random"), column=2)
        self._field(params, 1, "Зв’язок ієрархії:", self.linkage, ("ward", "complete", "average", "single"), column=0)
        self._field(params, 1, "DBSCAN eps:", self.eps, column=2)
        self._field(params, 2, "DBSCAN min_samples:", self.min_samples, column=0)
        self._field(params, 2, "Повторів вимірювання:", self.runs, column=2)
        self._field(params, 3, "Parallel K-Means перезапусків:", self.parallel_restarts, column=0)
        self._field(params, 3, "Parallel K-Means потоків:", self.parallel_workers, column=2)
        ttk.Checkbutton(params, text="Видаляти стоп-слова", variable=self.stopwords).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(params, text="Лематизувати українські слова", variable=self.lemmatize).grid(row=4, column=2, columnspan=2, sticky="w")
        ttk.Checkbutton(params, text="Створювати графіки", variable=self.charts).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(params, text="Рекомендувати K за silhouette", variable=self.recommend_k).grid(row=5, column=2, columnspan=2, sticky="w")
        ttk.Label(params, text="Власні стоп-слова через кому:").grid(row=6, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Entry(params, textvariable=self.custom_stopwords, width=24).grid(row=7, column=0, columnspan=4, sticky="ew")
        self.custom_stopwords.trace_add("write", lambda *_: self._update_stopword_count())
        self._update_stopword_count()
        ttk.Button(params, textvariable=self.stopword_count, command=self._show_stopwords).grid(row=8, column=0, columnspan=4, sticky="ew", pady=(5, 0))

        actions = ttk.Frame(self)
        actions.grid(row=2, column=0, sticky="ew", pady=10)
        self.run_button = ttk.Button(actions, text="Запустити експеримент", command=self._run)
        self.run_button.pack(side="left")
        ttk.Button(actions, text="Попередні експерименти…", command=self._browse_experiments).pack(side="left", padx=(8, 0))
        self.status = tk.StringVar(value="Оберіть файл і методи.")
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=14)

        results_box = ttk.LabelFrame(self, text="Результати та порівняння", padding=6)
        results_box.grid(row=3, column=0, sticky="nsew")
        results_box.columnconfigure(0, weight=1); results_box.rowconfigure(1, weight=1)
        self.summary = tk.StringVar(value="Після запуску тут буде показано найшвидший і найякісніший результат.")
        summary_row = ttk.Frame(results_box)
        summary_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        summary_row.columnconfigure(0, weight=1)
        ttk.Label(summary_row, textvariable=self.summary).grid(row=0, column=0, sticky="w")
        ttk.Button(summary_row, text="Порівняти результати", command=self._show_comparison).grid(row=0, column=1, sticky="e")
        columns = ("vectorizer", "algorithm", "clusters_found", "silhouette", "davies_bouldin",
                   "f1_macro", "f1_micro", "v_measure", "clustering_seconds", "clustering_memory_kb", "cpu_percent", "error")
        self.tree = ttk.Treeview(results_box, columns=columns, show="headings")
        headings = {"vectorizer": "Векторизація", "algorithm": "Алгоритм", "clusters_found": "Кластери",
                    "silhouette": "Silhouette", "davies_bouldin": "Davies–Bouldin", "f1_macro": "F1 macro",
                    "f1_micro": "F1 micro", "v_measure": "V-measure", "clustering_seconds": "Час, с",
                    "clustering_memory_kb": "Пам’ять, КБ", "cpu_percent": "CPU, %", "error": "Помилка"}
        for col in columns:
            self.tree.heading(col, text=headings[col]); self.tree.column(col, width=112 if col != "error" else 260, anchor="center")
        scroll = ttk.Scrollbar(results_box, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=scroll.set)
        self.tree.grid(row=1, column=0, sticky="nsew"); scroll.grid(row=2, column=0, sticky="ew")
        self.tree.bind("<Motion>", self._show_graph_on_hover)
        self.tree.bind("<Leave>", self._hide_graph_tip)
        self.tree.bind("<Double-1>", self._open_full_graph)
        self.artifacts = tk.StringVar()
        self.conclusion = tk.StringVar(value="")
        self.conclusion_title = ttk.Label(self, text="Загальний висновок:", font=("TkDefaultFont", 10, "bold"))
        self.conclusion_label = ttk.Label(self, textvariable=self.conclusion, style="Conclusion.TLabel", justify="left")
        self.artifacts_label = ttk.Label(self, textvariable=self.artifacts, foreground="#505050", justify="left")
        self.artifacts_label.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        self.bind("<Configure>", self._update_full_width_text)

    @staticmethod
    def _field(parent: ttk.Frame, row: int, label: str, value: tk.StringVar,
               choices: tuple[str, ...] | None = None, column: int = 0) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w")
        widget = ttk.Combobox(parent, textvariable=value, values=choices, state="readonly", width=13) if choices else ttk.Entry(parent, textvariable=value, width=15)
        widget.grid(row=row, column=column + 1, sticky="ew", padx=(6, 10), pady=1)

    def _choose_file(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("Text datasets", "*.txt *.csv *.json"), ("All files", "*.*")])
        if filename: self.path.set(filename)

    def _current_stopwords(self) -> list[str]:
        custom = {word.strip().lower() for word in self.custom_stopwords.get().split(",") if word.strip()}
        return sorted(set(StopwordRemover.UKRAINIAN_STOPWORDS) | custom)

    def _update_stopword_count(self) -> None:
        self.stopword_count.set(f"Переглянути стоп-слова ({len(self._current_stopwords())})")

    def _show_stopwords(self) -> None:
        """Display the base and current custom stop words before an experiment."""
        window = tk.Toplevel(self.root)
        window.title("Стоп-слова для поточного експерименту")
        window.geometry("410x470")
        window.transient(self.root)
        ttk.Label(window, text="Базові українські та додані користувачем стоп-слова.").pack(anchor="w", padx=12, pady=(12, 5))
        query = tk.StringVar()
        ttk.Entry(window, textvariable=query).pack(fill="x", padx=12)
        listbox = tk.Listbox(window, font=("TkFixedFont", 11))
        scroll = ttk.Scrollbar(window, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scroll.set)
        listbox.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=10)
        scroll.pack(side="right", fill="y", padx=(0, 12), pady=10)

        base = set(StopwordRemover.UKRAINIAN_STOPWORDS)
        custom = {word.strip().lower() for word in self.custom_stopwords.get().split(",") if word.strip()}
        words = self._current_stopwords()
        def refresh(*_args) -> None:
            needle = query.get().strip().lower()
            listbox.delete(0, "end")
            for word in words:
                if not needle or needle in word:
                    suffix = "  (додано)" if word in custom and word not in base else ""
                    listbox.insert("end", word + suffix)
        query.trace_add("write", refresh)
        refresh()

    def _run(self) -> None:
        try:
            selected_vectors = tuple(self.vector_list.get(i) for i in self.vector_list.curselection())
            selected_algorithms = tuple(self.algorithm_list.get(i) for i in self.algorithm_list.curselection())
            max_documents = int(self.max_documents.get()) if self.max_documents.get().strip() else None
            options = ExperimentOptions(input_path=self.path.get().strip(), text_column=self.text_column.get().strip(),
                label_column=self.label_column.get().strip(), max_documents=max_documents, remove_stopwords=self.stopwords.get(), lemmatize=self.lemmatize.get(),
                custom_stopwords=tuple(word.strip() for word in self.custom_stopwords.get().split(",") if word.strip()),
                vectorizers=selected_vectors, algorithms=selected_algorithms, n_clusters=int(self.n_clusters.get()),
                kmeans_init=self.kmeans_init.get(), linkage=self.linkage.get(), eps=float(self.eps.get()),
                min_samples=int(self.min_samples.get()), parallel_restarts=int(self.parallel_restarts.get()),
                parallel_workers=int(self.parallel_workers.get()), runs=int(self.runs.get()), make_charts=self.charts.get(),
                recommend_clusters=self.recommend_k.get())
            if not options.input_path: raise ValueError("Оберіть файл з текстами")
            if options.n_clusters < 2 or options.min_samples < 1 or options.parallel_restarts < 1 or options.parallel_workers < 1 or options.runs < 1 or (options.max_documents is not None and options.max_documents < 1): raise ValueError("Параметри мають бути додатними; кластерів — щонайменше 2")
        except ValueError as exc:
            messagebox.showerror("Перевірте параметри", str(exc)); return
        self.run_button.configure(state="disabled"); self.status.set("Виконується експеримент…")
        self.artifacts.set("")
        self._hide_conclusion()
        self.summary.set("Виконується експеримент…")
        self.row_chart_paths.clear()
        self.row_cpu_details.clear()
        for item in self.tree.get_children(): self.tree.delete(item)
        threading.Thread(target=self._worker, args=(options,), daemon=True).start()
        self.after(120, self._poll)

    def _worker(self, options: ExperimentOptions) -> None:
        try:
            result, artifacts = self.service.run(options, lambda text: self.events.put(("progress", text)))
            self.events.put(("complete", (result, artifacts)))
        except Exception as exc:
            self.events.put(("failure", str(exc)))

    def _poll(self) -> None:
        active = True
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress": self.status.set(str(payload))
                elif kind == "failure": messagebox.showerror("Помилка експерименту", str(payload)); active = False
                else:
                    results, artifacts = payload
                    self._populate_results(results, artifacts, "Готово")
                    active = False
        except queue.Empty: pass
        if active: self.after(120, self._poll)
        else: self.run_button.configure(state="normal")

    def _populate_results(self, results, artifacts: dict, status_prefix: str) -> None:
        self.data_section.collapse()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.row_chart_paths.clear()
        self.row_cpu_details.clear()
        for _, row in results.iterrows():
            values = []
            for column in self.tree["columns"]:
                value = row.get(column, "")
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            item_id = self.tree.insert("", "end", values=values)
            chart_key = f"chart_{row.get('vectorizer', '')}_{row.get('algorithm', '')}"
            if artifacts.get(chart_key):
                self.row_chart_paths[item_id] = artifacts[chart_key]
            self.row_cpu_details[item_id] = row.get("cpu_per_core_percent")
        self.status.set(f"{status_prefix}: {len(results)} комбінацій.")
        self._set_summary(results)
        if results.empty:
            self._hide_conclusion()
        else:
            self._show_conclusion(str(artifacts.get("conclusion", self.service.conclusion(results))))
        self.artifacts.set("Файли експерименту: " + ", ".join(
            f"{key}: {Path(value).name}" for key, value in artifacts.items()
            if key in {"csv", "json", "html", "comparison_chart", "manifest"}
        ))

    def _show_conclusion(self, text: str) -> None:
        self.conclusion.set(text)
        self.conclusion_title.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.conclusion_label.grid(row=5, column=0, sticky="ew")
        self._update_full_width_text()

    def _hide_conclusion(self) -> None:
        self.conclusion.set("")
        self.conclusion_title.grid_remove()
        self.conclusion_label.grid_remove()

    def _update_full_width_text(self, _event=None) -> None:
        """Keep narrative and file-path text readable across the whole window."""
        width = max(200, self.winfo_width() - 28)
        self.conclusion_label.configure(wraplength=width)
        self.artifacts_label.configure(wraplength=width)

    def _browse_experiments(self) -> None:
        root = self.service.output_dir / "experiments"
        folders = sorted((path for path in root.glob("experiment_*") if path.is_dir()), reverse=True)
        if not folders:
            messagebox.showinfo("Попередні експерименти", "Збережених експериментів ще немає.")
            return
        window = tk.Toplevel(self.root)
        window.title("Попередні експерименти")
        window.geometry("530x360")
        window.transient(self.root)
        ttk.Label(window, text="Оберіть експеримент, щоб переглянути його результати та графіки.").pack(anchor="w", padx=12, pady=(12, 6))
        listing = tk.Listbox(window, font=("TkFixedFont", 11))
        for folder in folders:
            listing.insert("end", folder.name.replace("experiment_", ""))
        listing.selection_set(0)
        listing.pack(fill="both", expand=True, padx=12, pady=6)
        def load_selected(_event=None) -> None:
            selected = listing.curselection()
            if not selected:
                return
            try:
                results, artifacts = self.service.load_saved_experiment(folders[selected[0]])
                self._populate_results(results, artifacts, "Відкрито експеримент")
                window.destroy()
            except Exception as exc:
                messagebox.showerror("Не вдалося відкрити експеримент", str(exc))
        controls = ttk.Frame(window)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(controls, text="Відкрити", command=load_selected).pack(side="right")
        ttk.Button(controls, text="Закрити", command=window.destroy).pack(side="right", padx=(0, 8))
        listing.bind("<Double-1>", load_selected)

    def _set_summary(self, results) -> None:
        """Present a practical quality and speed comparison without exports."""
        successful = results.dropna(subset=["clustering_seconds"]).copy() if "clustering_seconds" in results else results.iloc[0:0]
        if successful.empty:
            self.summary.set("Немає успішних результатів для порівняння.")
            return
        fastest = successful.loc[successful["clustering_seconds"].idxmin()]
        quality_column = "silhouette" if "silhouette" in successful else None
        best = successful.loc[successful[quality_column].idxmax()] if quality_column else fastest
        self.summary.set(
            f"Найшвидше: {fastest['vectorizer']} + {fastest['algorithm']} "
            f"({fastest['clustering_seconds']:.4f} с).  "
            f"Найкраща якість за Silhouette: {best['vectorizer']} + {best['algorithm']} "
            f"({best.get('silhouette', float('nan')):.4f}).  Наведіть курсор на рядок для графіка."
        )

    def _show_comparison(self) -> None:
        """Open a compact, sortable-at-a-glance comparison of completed runs."""
        items = self.tree.get_children()
        if not items:
            messagebox.showinfo("Порівняння", "Спершу запустіть хоча б один експеримент.")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Порівняння експериментів")
        dialog.geometry("980x390")
        ttk.Label(dialog, text="Вищий Silhouette означає компактніші кластери; нижчі час і пам’ять — ефективніше виконання.").pack(anchor="w", padx=12, pady=(12, 6))
        columns = ("Метод", "Алгоритм", "Silhouette", "Davies–Bouldin", "F1 macro", "V-measure", "Час, с", "Пам’ять, КБ", "CPU, %")
        table = ttk.Treeview(dialog, columns=columns, show="headings")
        for col in columns:
            table.heading(col, text=col)
            table.column(col, width=108, anchor="center")
        for item in items:
            values = self.tree.item(item, "values")
            # Corresponds to the main-table columns in the same order.
            row = (values[0], values[1], values[3], values[4], values[5], values[7], values[8], values[9], values[10])
            comparison_item = table.insert("", "end", values=row)
            table.set(comparison_item, "Метод", values[0])
            table.item(comparison_item, tags=(item,))
        scrollbar = ttk.Scrollbar(dialog, orient="horizontal", command=table.xview)
        table.configure(xscrollcommand=scrollbar.set)
        table.bind("<Motion>", lambda event: self._show_comparison_graph(event, table))
        table.bind("<Leave>", self._hide_graph_tip)
        table.bind("<Double-1>", lambda event: self._open_comparison_graph(event, table))
        table.pack(fill="both", expand=True, padx=12); scrollbar.pack(fill="x", padx=12, pady=(0, 12))

    def _show_graph_on_hover(self, event) -> None:
        item = self.tree.identify_row(event.y)
        if self.tree.identify_column(event.x) == "#11":
            marker = f"cpu:{item}"
            if marker == self._hovered_item:
                return
            self._hide_graph_tip()
            self._hovered_item = marker
            self._show_cpu_tip(self.tree.item(item, "values")[10] if item else "—", self.row_cpu_details.get(item))
            return
        if item == self._hovered_item:
            return
        self._hide_graph_tip()
        self._hovered_item = item
        path = self.row_chart_paths.get(item)
        self._show_chart_tip(path)

    def _show_comparison_graph(self, event, table: ttk.Treeview) -> None:
        item = table.identify_row(event.y)
        main_item = table.item(item, "tags")[0] if item and table.item(item, "tags") else ""
        if table.identify_column(event.x) == "#9":
            marker = f"comparison-cpu:{item}"
            if marker == self._hovered_item:
                return
            self._hide_graph_tip()
            self._hovered_item = marker
            self._show_cpu_tip(table.item(item, "values")[8] if item else "—", self.row_cpu_details.get(main_item))
            return
        marker = f"comparison:{item}"
        if marker == self._hovered_item:
            return
        self._hide_graph_tip()
        self._hovered_item = marker
        self._show_chart_tip(self.row_chart_paths.get(main_item))

    def _show_chart_tip(self, path: str | None) -> None:
        if not path or not Path(path).is_file():
            return
        try:
            image = tk.PhotoImage(file=path)
            # Larger preview for detailed inspection without opening the graph.
            divisor = max(1, (image.width() + 839) // 840, (image.height() + 559) // 560)
            if divisor > 1:
                image = image.subsample(divisor, divisor)
            tip = tk.Toplevel(self.root)
            tip.wm_overrideredirect(True)
            tip.attributes("-topmost", True)
            tip.geometry(f"+{self.root.winfo_pointerx() + 16}+{self.root.winfo_pointery() + 16}")
            ttk.Label(tip, image=image, text="").pack()
            self._graph_tip, self._tip_image = tip, image  # preserve image reference
        except tk.TclError:
            self._hide_graph_tip()

    def _show_cpu_tip(self, average: object, per_core: object) -> None:
        """Show process CPU average plus per-core system load for the task interval."""
        if isinstance(per_core, str):
            try:
                per_core = json.loads(per_core)
            except json.JSONDecodeError:
                per_core = None
        if not isinstance(per_core, (list, tuple)):
            return
        core_lines = []
        for start in range(0, len(per_core), 4):
            core_lines.append("   ".join(
                f"Ядро {index + 1}: {float(value):.1f}%"
                for index, value in enumerate(per_core[start:start + 4], start=start)
            ))
        text = (
            f"CPU процесу під час експерименту: {average}%\n"
            "Навантаження логічних ядер системи в той самий час:\n" +
            "\n".join(core_lines)
        )
        tip = tk.Toplevel(self.root)
        tip.wm_overrideredirect(True)
        tip.attributes("-topmost", True)
        ttk.Label(tip, text=text, justify="left", padding=10).pack()
        tip.update_idletasks()
        # Core details can be wide; show them on the left and keep the full
        # popup inside the current display.
        x = max(0, self.root.winfo_pointerx() - tip.winfo_reqwidth() - 16)
        y = min(self.root.winfo_pointery() + 16,
                self.root.winfo_screenheight() - tip.winfo_reqheight() - 8)
        tip.geometry(f"+{x}+{max(0, y)}")
        self._graph_tip = tip

    def _hide_graph_tip(self, _event=None) -> None:
        if self._graph_tip is not None:
            self._graph_tip.destroy()
        self._graph_tip, self._tip_image, self._hovered_item = None, None, ""

    def _open_full_graph(self, event) -> None:
        """Open the selected experiment's saved graph at its native size."""
        item = self.tree.identify_row(event.y)
        self._open_chart(self.row_chart_paths.get(item))

    def _open_comparison_graph(self, event, table: ttk.Treeview) -> None:
        item = table.identify_row(event.y)
        tags = table.item(item, "tags") if item else ()
        self._open_chart(self.row_chart_paths.get(tags[0]) if tags else None)

    def _open_chart(self, path: str | None) -> None:
        if not path or not Path(path).is_file():
            messagebox.showinfo("Графік", "Для цього результату графік не створено. Увімкніть «Створювати графіки» перед запуском.")
            return
        try:
            image = tk.PhotoImage(file=path)
            window = tk.Toplevel(self.root)
            window.title("Повний графік кластеризації")
            window.transient(self.root)
            ttk.Label(window, image=image).pack(padx=10, pady=10)
            window._chart_image = image  # keep the native-size image alive
        except tk.TclError as exc:
            messagebox.showerror("Графік", f"Не вдалося відкрити графік: {exc}")


def launch() -> None:
    root = tk.Tk()
    ClusteringApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
