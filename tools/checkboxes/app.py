from __future__ import annotations

import gradio as gr

from checkbox_pipeline import load_review_queue, save_and_next


with gr.Blocks(title="Checkbox Review Tool") as demo:
    gr.Markdown(
        "Upload a source JSON plus one or more scans. The app aligns each scan using the existing scan pipeline, crops checkbox regions from the source geometry, and lets you review them one by one."
    )

    with gr.Row():
        source_file = gr.File(label="Source JSON", file_types=[".json"])
        raw_files = gr.Files(label="Raw scans", file_types=["image", ".heic", ".heif"])

    load_button = gr.Button("Load crops", variant="primary")

    with gr.Row():
        crop_view = gr.Image(label="Checkbox crop", type="numpy", height=120)
        crop_label = gr.Markdown()

    mark_choice = gr.Radio(["Marked", "Unmarked"], value="Marked", label="Checkbox state")

    with gr.Row():
        save_button = gr.Button("Save selection", variant="primary")

    progress = gr.Markdown()

    samples_state = gr.State([])
    index_state = gr.State(0)
    saved_state = gr.State(0)

    load_button.click(
        load_review_queue,
        [source_file, raw_files],
        [samples_state, index_state, saved_state, crop_view, crop_label, progress, mark_choice],
    )

    save_button.click(
        save_and_next,
        [mark_choice, samples_state, index_state, saved_state],
        [crop_view, crop_label, progress, samples_state, index_state, saved_state, mark_choice],
    )


if __name__ == "__main__":
    demo.launch(share=True)
