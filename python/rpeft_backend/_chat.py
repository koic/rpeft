def render(tokenizer, messages, add_generation_prompt=False):
    """Renders a chat transcript into the text the model was trained to see.

    add_generation_prompt marks where the assistant's turn begins. It belongs to
    inference only: training renders the transcript as written.
    """
    _check_roles(messages)
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


def segments(tokenizer, messages):
    """Splits a rendered transcript into (text, trainable) pairs, trainable for the assistant's turns.

    Each assistant turn is located by rendering the transcript up to its generation prompt and up to the
    end of the turn: the difference is what the model writes at inference, the turn's end marker included.
    """
    rendered = render(tokenizer, messages)
    pairs = []
    position = 0
    for index, message in enumerate(messages):
        # A turn opening the transcript answers nothing, such as a greeting, so it is read as context.
        # Locating it would also mean rendering an empty transcript, which chat templates refuse.
        if index == 0 or message.get("role") != "assistant":
            continue

        start = render(tokenizer, messages[:index], add_generation_prompt=True)
        through = render(tokenizer, messages[: index + 1])
        # Some templates render earlier turns differently once the conversation goes on, such as dropping
        # reasoning from past replies. The turn then cannot be located, and guessing would train on the wrong text.
        if not (rendered.startswith(through) and through.startswith(start) and len(start) >= position):
            raise ValueError(
                "the chat template renders earlier turns differently as the conversation goes on, so the "
                "assistant's replies cannot be located; pass train_on_prompt: true (--train-on-prompt in the CLI) "
                "to train on the whole transcript"
            )

        pairs.append((rendered[position : len(start)], False))
        pairs.append((rendered[len(start) : len(through)], True))
        position = len(through)

    pairs.append((rendered[position:], False))
    return [(text, trainable) for text, trainable in pairs if text]


def writes_special_tokens(tokenizer):
    """Whether render's output already holds the special tokens, BOS included, that the model expects.

    A chat template writes them itself, so tokenizing its output must not add them again; the plain
    fallback rendering writes none.
    """
    return bool(getattr(tokenizer, "chat_template", None))


# Kept in step with RPEFT::Messages::ROLES. The Ruby side checks only the first line of a dataset file,
# so the rest of the file is checked here.
ROLES = ("system", "user", "assistant")


def _check_roles(messages):
    for message in messages:
        if message.get("role") not in ROLES:
            raise ValueError(
                "message role must be one of %s, got %r" % (", ".join(ROLES), message.get("role"))
            )


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
