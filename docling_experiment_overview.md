# Docling: Overview and Experiment Considerations (Stage 1.1)

## 1. Pipeline overview (standard PDF pipeline)

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 10, 'rankSpacing': 8, 'padding': 18, 'subGraphTitleMargin': {'top': 6, 'bottom': 22}}}}%%
flowchart LR
    subgraph build["build_pipe"]
        direction LR
        P(Preprocess) --> O(OCR) --> L(Layout) --> T(Tables) --> A(Assemble)
    end
    subgraph enrich["enrichment_pipe"]
        direction LR
        F(Formula) --> I(Image)
    end
    build --> enrich
    enrich --> DD[(DoclingDocument)]

    classDef b fill:#4a4a45,stroke:#a8a69c,color:#e8e6dc;
    classDef e fill:#085041,stroke:#1d9e75,color:#9fe1cb;
    classDef o fill:#3c3489,stroke:#7f77dd,color:#cecbf6;
    class P,L,T,A b;
    class O o;
    class F,I e;
    style build fill:none,stroke:#a8a69c,stroke-dasharray:4 4
    style enrich fill:none,stroke:#1d9e75,stroke-dasharray:4 4
```

The output of both pipes is the `DoclingDocument`, which feeds export and chunking. Grey: core stages. Purple: OCR, optional (`do_ocr`). Green: optional enrichment (Formula, Image), off by default.
Alternative: the **VLM pipeline** (e.g. Granite-Docling) replaces these stages with one vision-language model reading the page image. It is less configurable but a useful comparison point.

## 2. Customisation levers, by pipeline stage

🟢 consider for 1.1 · ⚪ later stage (greyed) · ⚫ out of scope for this text-quality batch, or hold fixed (greyed). Indented rows (↳) are values or sub-options of the option above. The PDF version greys out the ⚪ and ⚫ rows. Defaults and enums are from `docling/datamodel/pipeline_options.py` at commit `311b683`.

### Preprocess

Parses the PDF: text cells, geometry, page renders.

| Option | Values and what they do |
|:----------------------------------|:----------------------------------------------------------------------|
| 🟢 `PdfFormatOption(backend=…)` | `PdfBackend` enum. Compare if garbling persists. |
| ↳ 🟢 `PdfBackend.THREADED_DOCLING_PARSE, PdfBackend.PYPDFIUM2` | `THREADED_DOCLING_PARSE` (default, `"docling_parse"`): threaded Docling Parse, recommended for the standard pipeline. `PYPDFIUM2` (`"pypdfium2"`): fast, basic text extraction. |
| ↳ ⚫ `backend_options.parser_threads` | `ThreadedDoclingParseBackendOptions`. Parser worker threads, separate from `accelerator_options.num_threads`. |
| ↳ ⚫ `backend_options.max_concurrent_results` | Pages the backend decodes ahead of the pipeline. Bounds memory. |
| ⚫ `generate_page_images` | Default `False`. Attach a rendered image of each page. Needed for review views and `TableItem.get_image()`. |
| ⚪ `images_scale` | Default `1.0`. Scale of generated images (`2.0` high resolution, `0.5` previews). Stage 1.5. |
| ⚫ `accelerator_options` | Hold fixed across runs. |
| ↳ ⚫ `device` | `auto` (default), `cpu`, `cuda`, `cuda:N`, `mps`, `xpu`. |
| ↳ ⚫ `num_threads` | Default `4`. CPU threads for model inference. |
| ↳ ⚫ `cuda_use_flash_attention2` | Default `False`. |

### OCR

Recognises text from page images, merged with text-layer cells (optional).

| Option | Values and what they do |
|:----------------------------------|:----------------------------------------------------------------------|
| 🟢 `do_ocr` | Default `True`. Set `False` for 1.1 baselines; scans then give empty text. |
| ⚪ `ocr_options` | Default `OcrAutoOptions()`. Engine kinds: `auto`, `rapidocr`, `easyocr`, `tesseract` (CLI), `tesserocr`, `ocrmac`, `nemotron-ocr`, `kserve_v2_ocr`. Stage 1.3. |
| ↳ ⚪ `auto` | Probes in order: OcrMac, Nemotron OCR (Linux with CUDA), RapidOCR (onnxruntime), EasyOCR, RapidOCR (torch). Never picks Tesseract. |
| ↳ ⚪ `mode` | `OcrMode`: `DEFAULT` (wired to `PDF_AWARE_LAYOUT_REGIONS`), `FULL_PAGE`, `LAYOUT_REGIONS` (layout detections only), `PDF_AWARE_LAYOUT_REGIONS` (drops clusters that hold only text PDF cells). |
| ↳ 🟢 `replace_broken_text_layer` | Default `False`. `True` re-reads pages whose text layer is broken (e.g. unmapped glyphs) with full-page OCR, per page. Relevant to garbled extractions. Tested in 1.1 (E2). |
| ↳ ⚪ `lang` | Default `[]`: engine default. Native engine codes, or `iso:` BCP-47 tags. |
| ↳ ⚪ `scale` | Default `3.0` (216 DPI). Lower it when the source image is already high resolution. |
| ↳ ⚪ `rapidocr` | `backend`: `onnxruntime` (default), `openvino`, `paddle`, `torch`. `model_size` `small`, `text_score` `0.5`. |
| ↳ ⚪ `easyocr` | `recog_network` `standard` (default) or `craft`, `confidence_threshold` `0.5`, `use_gpu` auto. |
| ↳ ⚪ `tesseract, tesserocr` | `psm`, `path`; CLI also `tesseract_cmd`. Default `lang` `[eng, spa, fra, deu]`. |
| ↳ ⚪ `ocrmac` | `recognition` `accurate` (default) or `fast`; `framework` `vision`. |
| ⚪ `ocr_batch_size` | Default `4`. Stage 1.7. |

### Layout

Detects boxes and labels (text, table, figure, heading).

| Option | Values and what they do |
|:----------------------------------|:----------------------------------------------------------------------|
| 🟢 `layout_options` | Default `LayoutObjectDetectionOptions`, preset `layout_heron_default`. Main layout variable. External: domain-tuned layout model via plugin; rule-based header, footer and disclaimer filters (1.4). |
| ↳ 🟢 `preset` | `layout_heron_default` (default), `layout_heron_101`, `layout_egret_medium`, `layout_egret_large`, `layout_egret_xlarge` (higher accuracy on complex documents). Preset ids for 101 and medium inferred from the model specs. |
| ↳ ⚫ `keep_empty_clusters` | Default `False`. No effect on text. |
| ↳ ⚫ `skip_cell_assignment` | Default `False`. `True` skips assigning cells to clusters, for speed; likely drops text. |
| ↳ 🟢 `create_orphan_clusters` | Default `True`. Isolated elements get their own cluster. |
| ↳ 🟢 `engine_options.score_threshold` | Default `0.3` (from the docs). Minimum detection confidence. |
| ↳ 🟢 `run_postprocessor` | Default `True`. `False` passes raw clusters through. |
| 🟢 `force_backend_text` | Default `False`. `True` re-reads each element's text from the PDF backend inside its box, instead of joining the text cells assigned to it (which may include OCR cells). With OCR off the difference is likely small; verify in the source. External: repair ligature or encoding issues after extraction. |
| ⚪ `layout_batch_size` | Default `4`. Stage 1.7. |

### Tables

Rebuilds row and column structure.

| Option | Values and what they do |
|:----------------------------------|:----------------------------------------------------------------------|
| 🟢 `do_table_structure` | Default `True`. `False` still detects tables (label `TABLE`) but skips rebuilding rows and columns. Table text is excluded from the text-only output, so likely fixed at `False` after a check (E6). |
| ⚫ `table_structure_options` | Default `TableStructureOptions()` (TableFormer V1). Applies only when `do_table_structure=True`. Compare the three classes. External: alternative table extractors for hard tables. |
| ↳ ⚫ `TableStructureOptions.mode` | `TableFormerMode`: `ACCURATE` (default) or `FAST`. |
| ↳ ⚫ `TableStructureOptions.do_cell_matching` | Default `True`. Matches predictions back to PDF cells. |
| ↳ ⚫ `TableStructureV2Options` | TableFormer V2. `do_cell_matching` default `True`; can break tables when PDF cells merge across columns. `False` lets the model define cells and ignores PDF cells. |
| ↳ ⚫ `GraniteVisionTableStructureOptions` | VLM-based. `scale` default `2.0` (1.0 = 72 dpi); small table text needs a higher scale. |
| ⚫ `table_batch_size` | Default `4`. |

### Assemble

Builds the document: reading order, heading levels, scripts, attached images.

| Option | Values and what they do |
|:----------------------------------|:----------------------------------------------------------------------|
| ⚪ `heading_hierarchy_options` | Default `enabled=False`: every heading stays level 1. Runs right after the reading-order model. Needed for hierarchical chunks (1.6). External: custom heading rules for house report formats. Stage 1.6. |
| ↳ ⚪ `use_bookmarks` | Default `True`. PDF outline is authoritative when it matches a heading. |
| ↳ ⚪ `use_numbering` | Default `True`. Legal or outline numbering. |
| ↳ ⚪ `use_style` | Default `True`. Heading font size as a fallback. |
| ↳ ⚪ `use_font_style` | Default `True`. Weight, slant and all-caps refine the style fallback. |
| ↳ ⚫ `style_size_tolerance` | Default `0.05`. |
| ↳ ⚫ `bookmark_match_threshold` | Default `0.8`. Minimum title similarity. |
| ↳ ⚫ `max_level` | Default `6`. |
| ↳ ⚫ `numbering_schemes, numbering_keywords, numbering_number_words` | Default `None`, `{}`, `[]`. Overrides for non-English numbering. |
| ⚫ `script_position_options` | Default `enabled=False`. Recovers superscripts and subscripts as inline runs with `formatting.script`. Reads character positions with pypdfium2; skipped if missing. Leaves headings, captions, tables, OCR text and invisible text unchanged. External: post-process footnote links. |
| 🟢 `use_reading_order_separators` | Default `True`. Uses visible rules as reading-order signals; only for backends that expose shape geometry. External: learned reading-order model if layouts stay problematic. |
| ⚫ `generate_parsed_pages` | Default `False`. Keeps parsed pages for debugging and `ConversionResult.pages`. Uses more memory; dropped after assembly unless set. |
| ⚫ `generate_picture_images` | Default `False`. Extract embedded images and attach them to picture items. |
| ⚫ `use_embedded_picture_images` | Default `False`. Use the PDF's embedded image when a picture region holds exactly one safe image; else the rendered page crop. |

### Formula

Recognises formulas and code (optional enrichment).

| Option | Values and what they do |
|:----------------------------------|:----------------------------------------------------------------------|
| ⚫ `do_formula_enrichment` | Default `False`. |
| ⚫ `do_code_enrichment` | Default `False`. |
| ⚫ `code_formula_options` | Default preset `default`. `extract_code`, `extract_formulas` `True`; `scale` `2.0`; `expansion_factor` `0.18`. |

### Image

Classifies pictures, captions them and converts charts to data (optional enrichment).

| Option | Values and what they do |
|:----------------------------------|:----------------------------------------------------------------------|
| ⚪ `do_picture_classification` | Default `False`. Stage 1.5. |
| ⚪ `picture_classification_options` | Engine and model; `top_k` default `None`. Stage 1.5. |
| ⚪ `do_picture_description` | Default `False`. Stage 1.5. |
| ⚪ `picture_description_options` | Default preset `smolvlm` (`granite_vision` available). `batch_size` `8`, `scale` `2.0`, `picture_area_threshold` `0.05`, `classification_allow`, `classification_deny`, `classification_min_confidence` `0.0`. API mode needs `enable_remote_services=True`. Stage 1.5. |
| ⚪ `do_chart_extraction` | Default `False`. Charts into tables; turns on picture classification. Stage 1.5. |
| ⚪ `chart_extraction_options` | Default preset `granite_vision_v4` (`granite_vision_v3_3` is CSV-only). `chart2csv` `True`, `chart2summary` `False`, `chart2code` `False`, `scale` `2.0`. Stage 1.5. |

### Whole pipeline

Settings that span stages.

| Option | Values and what they do |
|:----------------------------------|:----------------------------------------------------------------------|
| 🟢 `ProcessingPipeline` | Enum: `STANDARD`, `NATIVE`, `VLM`, `LEGACY`, `ASR`. `STANDARD` is the only pipeline used (use `ThreadedPdfPipelineOptions`, which adds nothing to `PdfPipelineOptions`). |
| ↳ 🟢 `STANDARD` | General document pipeline with layout analysis. Baseline. |
| ↳ ⚫ `NATIVE` | Model-free: docling-parse text cells and embedded images only, no reading order, headings or tables. Fast text-layer baseline for garbling checks. |
| ↳ ⚫ `VLM` | One vision-language model reads the page image (default preset `granite_docling`). Comparison baseline. |
| ↳ ⚫ `LEGACY, ASR` | Backward compatibility; speech recognition. Not relevant. |
| ⚫ `NativePdfPipelineOptions` | For `NATIVE`: `text_cell_unit` `LINE` (default), word or char; `parser_threads` default all but one CPU thread. |
| ⚫ `document_timeout` | Default `None`. Seconds before returning partial results (`PARTIAL_SUCCESS`, timeout error). Set 90–120 s for batch runs. |
| ⚫ `artifacts_path` | Default `None`: models download on first use. Pin a local directory for reproducibility. |
| ⚪ `queue_max_size` | Default `100`. Items buffered between threaded stages. Stage 1.7. |
| ⚫ `enable_remote_services` | Default `False`. |
| ⚫ `allow_external_plugins` | Default `False`. |

### Ignore (not relevant to this task)

| Option | What it does | Why ignore |
|:----------------------------|:--------------------------------------------|:--------------------------|
| ⚫ `extract_form_fields` | Default `False`. Extracts native fillable form widgets | Research PDFs are not forms |
| ⚫ `do_media_conversion`, `keep_media`, `media_asr_options` | Default `False`, `False`, Whisper tiny. Audio and video embedded in slides | Applies to PPTX and ODP, not PDF |
| ⚫ `batch_polling_interval_seconds` | Default `0.5`. Threaded stage polling interval | Internal plumbing |
| ⚫ `stage_shutdown_timeout_seconds` | Default `15.0`. Wait for stage threads to stop | Internal plumbing |
| ⚫ `generate_table_images` | Deprecated. Use `generate_page_images=True` and `TableItem.get_image()` | Deprecated |
| ⚫ `force_full_page_ocr` | Deprecated. Use `ocr_options.mode=OcrMode.FULL_PAGE` | Deprecated |
| ⚫ `kind` | Class-level discriminator | Not a user setting |

Stage placement of a few options is approximate. Check the Docling source for the exact stage.

## 3. Experiment plan (stage 1.1)

**Aim:** the best text-only extraction. A custom serializer keeps body text (`TEXT`, `PARAGRAPH`, `SECTION_HEADER`, `TITLE`, `LIST_ITEM`) and drops tables, pictures, captions, footnotes, headers and footers. Layout levers matter only through their effect on that text. Experiments cover only the 🟢 levers in section 2. OCR is limited to `replace_broken_text_layer` on garbled pages; engine and mode choice wait for 1.3.

### What is measured

| Target | Measured on | Question | Metrics |
|:-------|:------------|:---------|:--------|
| **A. Text quality** | All output text, before filtering | Are the characters and words right? | Garble rate, dictionary hit rate, dropped-text rate (no gold needed); diacritic fidelity (gold) |
| **C. Serializer output** | Text after filtering | Does it match gold body text, without table or figure text mixed in? | Loss rate, leak rate, separation rate, order gap |

**Primary metrics:** loss rate and garble rate. Leak rate is secondary, with a maximum allowed value. Separation rate shows the structural gain over `pypdf`.

### Metric definitions

All text is normalised identically first: NFC, whitespace collapsed, line-end hyphenation joined, Markdown stripped. Word overlap counts words as a multiset, order ignored.

| Metric | Definition | Gold? |
|:-------|:-----------|:------|
| Garble rate | Share of words containing `U+FFFD`, `(cid:n)`, private-use characters, broken ligatures or detached diacritics (e.g. a spacing `¨` beside its letter) | No |
| Dictionary hit rate | Share of words found in an English word list, extended with accented names (e.g. Türkiye, São Paulo, Côte d'Ivoire) | No |
| Diacritic fidelity | Share of gold words with accented characters (e.g. Türkiye) reproduced exactly after NFC: not stripped (Turkiye), split (T¨ urkiye) or replaced | Body |
| Dropped-text rate | Share of text-layer words (raw docling-parse cells) that appear in no output item | No |
| Loss rate | Share of gold body words missing from the output (1 − word recall) | Body |
| Leak rate | Share of output words not in gold body text. Split by source: gold table and figure text, other (headers, footers, footnotes, captions, noise) | Body, table and figure |
| Separation rate | Share of gold table and figure words kept out of the output. `pypdf` scores near 0; this is the structural gain | Table and figure |
| Leak source | For Docling, the label of the item each leaked word came from (e.g. `PAGE_FOOTER` text labelled `TEXT`). Diagnostic only | No |
| Order gap | Order-insensitive word F1 minus order-sensitive score (1 − normalised word edit distance) | Body |

### Reference rows

- **P0:** `pypdf`, its full page text taken as output. Shown in every results table.
- **B0:** `STANDARD`, defaults except `do_ocr=False`. Run twice as a determinism check.

### Experiments

Run in order. Each changes one lever against the current best config (B0 until a winner is found).

| # | Stage | Lever | Variants (default first) | Target | Primary metrics |
|:--|:------|:------|:-------------------------|:-------|:----------------|
| E1 | Preprocess | `PdfBackend` | `THREADED_DOCLING_PARSE`, `PYPDFIUM2` | A | Garble rate, dictionary hit rate, dropped-text rate, diacritic fidelity |
| E2 | OCR, Layout | `do_ocr` × `force_backend_text`, with `replace_broken_text_layer=True` when OCR is on | 2×2 grid; garbled-page stratum reported separately | A | Garble rate, loss rate, `from_ocr` fraction, time per page |
| E3 | Layout | `layout_options.preset` | `layout_heron_default`, `layout_heron_101`, `layout_egret_medium`, `layout_egret_large`, `layout_egret_xlarge` | C | Loss rate, leak rate, separation rate, latency |
| E4 | Layout | `engine_options.score_threshold` (verify path in source) | 0.2, 0.3, 0.4, 0.5 on the E3 winner | C | Loss against leak trade-off |
| E5 | Layout | `create_orphan_clusters`; `run_postprocessor` | Each toggled once, separately | A, C | Dropped-text rate, loss rate, leak rate |
| E6 | Tables | `do_table_structure` | `True`, `False` | Cost | Confirm loss, leak and separation unchanged (table text must not fall back into body), then fix at `False` for speed |
| E7 | Assemble | `use_reading_order_separators` | `True`, `False` (no effect with `PYPDFIUM2`) | C | Order gap |
| E8 | Combine | Levers whose effect beat the noise floor | Full grid if 3 or fewer levers, else add one at a time | A, C | All primary metrics, interaction effects |
| E9 | Validate | Winner and runner-up | Full corpus, plus the 20 held-out documents | A, C | Label-free scores, failure rate, time per page |

Order: E1 → E2 (text source fixed) → E3 → E4 → E5 (layout) → E6, E7 → E8 → E9. Heading hierarchy is deferred to 1.6.

### Judging differences

- **Unit:** pages, grouped by stratum (below), so a regression in one type is not hidden by the mean.
- **Comparison:** pair every config with the current best on the same pages. Report the mean difference with a bootstrap 95% interval, resampling by document, plus win, tie and loss counts.
- **Noise floor:** CPU inference is near-deterministic, so the B0 rerun is a determinism check. Uncertainty comes from page sampling, which the bootstrap covers.
- **Overfitting:** E1 to E7 are screened on 80 documents; confirm winners on the 20 held-out documents, untouched until E9.
- **Decision rule, written before running:** loss and garble rates improve with an interval that excludes zero, leak rate stays under its maximum, no stratum regresses by more than a set margin, and time per page stays under a fixed budget (set on fixed hardware before E1). Choose from the quality-against-time Pareto front.

### Sample and gold labels

- **Sample:** 100 documents, stratified by layout type: multi-column, table-heavy, chart-heavy, footnote-heavy, boilerplate-heavy, accented names or special characters (e.g. Türkiye), and documents where `pypdf` garbles (oversampled). A document can count towards several strata; tag each page with its stratum. Split 80 for screening, 20 held out. Bootstrap intervals resample by document.
- **Label-free metrics** (garble, dictionary, dropped text) run on every page of all 100 documents.
- **Gold per page, two text fields:** (1) body text, matching the serializer keep set; (2) table and figure text (cell contents, chart labels and axis text), unordered. Gold on 2 pages per document (about 200 pages), chosen to cover each document's strata. Draft from the page image with Claude, not from a Docling run, to avoid biasing gold toward B0; then correct by hand. Field 2 needs no ordering, so correction is quick.
- **No layout boxes, no docling-eval:** layout is judged only through the text.

## 4. Experiment hygiene

- Each config is a small named, immutable object. Log it, with metrics and Markdown outputs, to MLflow.
- Cache outputs by (PDF hash, config hash) so only changed configs re-run.
- Evaluate in layers: intrinsic checks on the full corpus, a gold set of about 200 pages from 100 documents, then downstream tasks (retrieval, tagging).
- Verify defaults against the installed version with `PdfPipelineOptions().model_dump()` before fixing the baseline.
