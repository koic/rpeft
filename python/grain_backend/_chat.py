def render(tokenizer, messages, add_generation_prompt=False):
    """Renders a chat transcript into the text the model was trained to see.

    add_generation_prompt marks where the assistant's turn begins. It belongs to
    inference only: training renders the transcript as written.
    """
    messages = _without_control_tokens(tokenizer, messages)
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


def _without_control_tokens(tokenizer, messages):
    # The rendered text is tokenized as a whole, so a control token such as <|im_end|> written inside
    # a message would end the turn there and the rest would be read as a turn of its own. Message text
    # is data, so such tokens are removed from it; only the chat template may produce them.
    tokens = _control_tokens(tokenizer)
    if not tokens:
        return messages

    return [
        dict(message, content=_strip(message["content"], tokens))
        if isinstance(message.get("content"), str)
        else message
        for message in messages
    ]


def _control_tokens(tokenizer):
    tokens = set(getattr(tokenizer, "all_special_tokens", None) or [])
    for token in (getattr(tokenizer, "added_tokens_decoder", None) or {}).values():
        if getattr(token, "special", False):
            tokens.add(token.content)
    return sorted(tokens, key=len, reverse=True)


def _strip(content, tokens):
    # Repeated until stable, so removing one token cannot join its neighbors into another.
    while True:
        stripped = content
        for token in tokens:
            stripped = stripped.replace(token, "")
        if stripped == content:
            return stripped
        content = stripped
