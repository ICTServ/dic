from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr


DEBUG_UI = False
LANGS = {
    'English': 'eng_Latn',
    'Interslavic': 'isv_Latn',
    # 'Интерславик': 'isv_Cyrl',
    'Russian': 'rus_Cyrl',
    'Belarusian': 'bel_Cyrl',
    'Ukrainian': 'ukr_Cyrl',
    'Polish': 'pol_Latn',
    'Silesian': 'szl_Latn',
    'Czech': 'ces_Latn',
    'Slovak': 'slk_Latn',
    'Slovenian': 'slv_Latn',
    'Croatian': 'hrv_Latn',
    'Bosnian': 'bos_Latn',
    'Serbian': 'srp_Cyrl',
    'Macedonian': 'mkd_Cyrl',
    'Bulgarian': 'bul_Cyrl',
    'Esperanto': 'epo_Latn',
    'German': 'deu_Latn',
    'French': 'fra_Latn',
    'Spanish': 'spa_Latn',
}


if DEBUG_UI:
    def translate(text, src_lang, tgt_lang):
        return text    

else:
    model_name = 'Salavat/nllb-200-distilled-600M-finetuned-isv_v2'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lang2id = tokenizer.lang_code_to_id
    lang2id['isv_Latn'] = tokenizer(['isv_Latn'], add_special_tokens=False)['input_ids'][0][0]
    lang2id['isv_Cyrl'] = tokenizer(['isv_Cyrl'], add_special_tokens=False)['input_ids'][0][0]
            
    def translate(text, from_, to_):
        # empty line hallucinations fix
        lines = [f'{line} ' for line in text.split('\n')] if text else ''
        inputs = tokenizer(lines, return_tensors="pt", padding=True)
        inputs['input_ids'][:, 0] = lang2id[LANGS[from_]]
        translated_tokens = model.generate(**inputs, max_length=400, forced_bos_token_id=lang2id[LANGS[to_]])
        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        output = '\n'.join(result)
        return output
        

with gr.Blocks() as demo:
    gr.Markdown('<div align="center"><h1>Interslavic translator</h1></div>')
    with gr.Row():
        lang_input = gr.components.Dropdown(label="From", choices=list(LANGS.keys()), value='English')
        lang_output = gr.components.Dropdown(label="To", choices=list(LANGS.keys()), value='Interslavic')
    with gr.Row().style(equal_height=True):
        text_input = gr.components.Textbox(label="Text", lines=5, placeholder="Your text")
        text_output = gr.components.Textbox(label="Result", lines=5, placeholder="Translation...")
    translate_btn = gr.Button("Translate")
    gr.Markdown((
        'Finetuned model [NLLB200](https://ai.facebook.com/research/no-language-left-behind/) '
        'using corpus of [Inter-Slavic](https://interslavic-dictionary.com/grammar) language'
    ))
    translate_btn.click(translate, inputs=[text_input, lang_input, lang_output], outputs=text_output)

demo.launch()
