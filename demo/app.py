import gradio as gr
import os

from wavlmmsdd.audio.utils.utils import Build
from wavlmmsdd.audio.feature.embedding import WavLMSV
from wavlmmsdd.audio.preprocess.convert import Convert
from wavlmmsdd.audio.preprocess.resample import Resample
from wavlmmsdd.audio.diarization.diarize import Diarizer

def diarize_audio(audio_file):

    if audio_file is None:
        return "Lütfen bir ses dosyası yükleyin!"

    input_path = audio_file.name

    resampler = Resample(audio_file=input_path)
    wave_16k, sr_16k = resampler.to_16k()

    converter = Convert(waveform=wave_16k, sample_rate=sr_16k)
    converter.to_mono()
    saved_path = converter.save()

    builder = Build(saved_path)
    manifest_path = builder.manifest()

    embedder = WavLMSV()

    diarizer = Diarizer(embedding=embedder, manifest_path=manifest_path)
    results = diarizer.run()

    return str(results)


def build_interface():
    with gr.Blocks():
        gr.Markdown(
            "# WavLM + MSDD Diarization Demo\nBu arayüzle ses dosyası yükleyip diyarezasyon sonucu alabilirsiniz.")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="Ses Yükle (WAV/MP3)", type="file")
                diarize_button = gr.Button("Diyarezasyon Başlat")

            with gr.Column():
                output_text = gr.Textbox(label="Diyarezasyon Çıktısı")

        diarize_button.click(
            fn=diarize_audio,
            inputs=[audio_input],
            outputs=[output_text]
        )

    return gr.Blocks.get_current()


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
