def render(tokenizer, messages, add_generation_prompt=False):
    """Renders a chat transcript into the text the model was trained to see.

    add_generation_prompt marks where the assistant's turn begins. It belongs to
    inference only: training renders the transcript as written.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    # Tokenizers without a chat template (small test models, some base models) still need
    # a deterministic rendering.
    rendered = "\n".join(
        "%s: %s" % (message["role"], message["content"]) for message in messages
    )
    if add_generation_prompt:
        rendered += "\nassistant:"

    return rendered
