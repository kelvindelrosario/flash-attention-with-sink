# flash attention with sink

This repo is trying to implement the attention with sink in [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b).

To test, run:

```bash
python test.py
```

⚠️ The backward implementation within this repo is not correct at the moment.

## TODO

- [ ] backward
- [ ] varlen version
- [ ] communicate with flash attention community to only return `softmax_lse` but not `S_dmask`.
