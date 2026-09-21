"""Выполнить весь ноутбук и сохранить HTML с кодом и результатами."""
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / 'statistics_project.ipynb'
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=600, kernel_name='python3',
               resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(nb, path)
exporter = HTMLExporter(template_name='lab')
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
body, _ = exporter.from_notebook_node(nb)
(root / 'statistics_report.html').write_text(body, encoding='utf-8')
print('Notebook executed; HTML exported.')
